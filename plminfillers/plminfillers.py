import ablang2
import torch
import os
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForMaskedLM


AA = "ACDEFGHIKLMNPQRSTVWY"


class PLMInfiller:
    """
    Unified wrapper around HuggingFace MLM and ablang2 protein language models.

    Supports: VHHBERT, nanoBERT, AntiBERTa2, AbLang, AbLang2, plus any
    local Hugging Face-style models discovered under kuzco_models/*/final_model.
    """

    _BASE_DIR = Path(__file__).resolve().parent
    _KUZCO_MODELS_DIR = Path(
        os.environ.get("KUZCO_MODELS_DIR", str(_BASE_DIR / "kuzco_models"))
    )

    # Models whose HF tokenizer needs space-separated single-char input
    _SPACE_SEP = {"VHHBERT", "AntiBERTa2"}

    _MODEL_GENERATORS = {
        "VHHBERT": lambda: AutoModelForMaskedLM.from_pretrained("COGNANO/VHHBERT"),
        "nanoBERT": lambda: AutoModelForMaskedLM.from_pretrained("NaturalAntibody/nanoBERT"),
        "AntiBERTa2": lambda: AutoModelForMaskedLM.from_pretrained("alchemab/antiberta2"),
        "AbLang": lambda device: ablang2.pretrained(model_to_use="ablang1-heavy", random_init=False, device=device),
        "AbLang2": lambda device: ablang2.pretrained(model_to_use="ablang2-paired", random_init=False, device=device),
    }

    _TOKENIZER_GENERATORS = {
        "VHHBERT": lambda: AutoTokenizer.from_pretrained("COGNANO/VHHBERT"),
        "nanoBERT": lambda: AutoTokenizer.from_pretrained("NaturalAntibody/nanoBERT"),
        "AntiBERTa2": lambda: AutoTokenizer.from_pretrained("alchemab/antiberta2"),
    }

    @classmethod
    def _discover_local_models(cls) -> dict[str, Path]:
        discovered_models = {}
        if not cls._KUZCO_MODELS_DIR.exists():
            return discovered_models

        for model_dir in sorted(cls._KUZCO_MODELS_DIR.iterdir()):
            if not model_dir.is_dir():
                continue
            final_model_dir = model_dir / "final_model"
            if final_model_dir.is_dir():
                discovered_models[model_dir.name] = final_model_dir

        return discovered_models

    @classmethod
    def _build_model_generators(cls) -> dict[str, callable]:
        model_generators = dict(cls._MODEL_GENERATORS)
        for model_name, model_path in cls._discover_local_models().items():
            model_generators[model_name] = lambda model_path=model_path: AutoModelForMaskedLM.from_pretrained(model_path)
        return model_generators

    @classmethod
    def _build_tokenizer_generators(cls) -> dict[str, callable]:
        tokenizer_generators = dict(cls._TOKENIZER_GENERATORS)
        for model_name, model_path in cls._discover_local_models().items():
            tokenizer_generators[model_name] = lambda model_path=model_path: AutoTokenizer.from_pretrained(model_path)
        return tokenizer_generators

    _ALL_MODEL_GENERATORS = {}
    _ALL_TOKENIZER_GENERATORS = {}
    SUPPORTED_MODELS = set()

    def __init__(self, name: str, device: str = None):
        if name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model name: {name}")

        self.name = name
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._is_ablang = name in ("AbLang", "AbLang2")

        if self._is_ablang:
            self._model = self._MODEL_GENERATORS[name](self.device)
            self._tokenizer = self._model.tokenizer
            vocab = self._tokenizer.vocab_to_token if name == "AbLang" else self._tokenizer.aa_to_token
            self._aa_ids = {aa: vocab[aa] for aa in AA}
            self._mask_id = vocab["*"]
            self._specials = set(self._tokenizer.all_special_tokens) if name == "AbLang2" else None
        else:
            self._model = self._ALL_MODEL_GENERATORS[name]().to(self.device).eval()
            self._tokenizer = self._ALL_TOKENIZER_GENERATORS[name]()
            self._aa_ids = {aa: self._tokenizer.convert_tokens_to_ids(aa) for aa in AA}
            self._mask_id = self._tokenizer.mask_token_id
            self._special_ids = {
                self._tokenizer.cls_token_id, self._tokenizer.sep_token_id, self._tokenizer.pad_token_id
            } - {None}

    ###
    # internal helpers
    ###
    def _tokenize(self, seq: str):
        """Return (token_ids [1,L], list_of_residue_positions)."""
        if self._is_ablang:
            tok = self._tokenizer
            if self.name == "AbLang":
                ids = tok([seq], pad=True, device=self.device)
                res = list(range(1, ids.shape[1] - 1))
            else:
                ids = tok([f"<{seq}>|"], pad=True, w_extra_tkns=False, device=self.device)
                res = [i for i, t in enumerate(ids[0].tolist()) if t not in self._specials]
            return ids, res

        prepared_seq = " ".join(seq) if self.name in self._SPACE_SEP else seq
        enc = self._tokenizer(prepared_seq, return_tensors="pt")
        ids = enc["input_ids"]
        res = [i for i, t in enumerate(ids[0].tolist()) if t not in self._special_ids]
        return ids, res

    def _forward(self, token_ids):
        """Return raw logits tensor [1, seq_len, vocab_size]."""
        if self._is_ablang:
            return self._model.AbLang(token_ids)
        return self._model(token_ids.to(self.device)).logits

    ###
    # public API
    ###
    @torch.no_grad()
    def infill(self, seq: str, positions: list[int]) -> str:
        """Predict residues at the given 0-based positions. Returns AA string."""
        per_pos = self.infill_logits(seq, positions)
        return "".join(max(d, key=d.get) for d in per_pos)

    @torch.no_grad()
    def infill_logits(self, seq: str, positions: list[int]) -> list[dict[str, float]]:
        """Return raw logits at specified 0-based residue positions.

        Replaces those positions with [MASK] and does a single forward pass.
        Returns [{AA: logit}, …] in the same order as `positions`.
        """
        tokens, res_pos = self._tokenize(seq)
        token_positions = [res_pos[p] for p in positions]

        masked = tokens.clone()
        for tp in token_positions:
            masked[0, tp] = self._mask_id

        logits = self._forward(masked)

        return [
            {aa: logits[0, tp, tid].item() for aa, tid in self._aa_ids.items()}
            for tp in token_positions
        ]


PLMInfiller._ALL_MODEL_GENERATORS = PLMInfiller._build_model_generators()
PLMInfiller._ALL_TOKENIZER_GENERATORS = PLMInfiller._build_tokenizer_generators()
PLMInfiller.SUPPORTED_MODELS = set(PLMInfiller._ALL_MODEL_GENERATORS.keys())
