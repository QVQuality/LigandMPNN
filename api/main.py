import logging
import os
from pathlib import Path
import shutil
import subprocess as sp
import tempfile
import time
from typing import Optional

from Bio import SeqIO
from fastapi import FastAPI, File, Form, UploadFile, HTTPException


BASE_DIR = Path("/outputs")

logging.basicConfig(level=logging.INFO)

app = FastAPI(title="Unified RosettaCommons ProteinMPNN-based models API")


@app.get("/v1/health/ready")
async def health_check():
    return {"status": "ready"}


async def paint_with_an_rcmpnn(
    model_type: str,
    pdb_file: UploadFile,
    designed_positions: str,
    num_seq_per_target: int,
    omit_aas: str | None,
    temperature: float,
    random_seed: int,
    ckpt_path: Path | None,
    run_output_dirname: str,
    plm_bias: str = "",
    plm_bias_mode: str = "aware",
    plm_bias_weight: float = 1.0,
    save_stats: bool = False,
):
    # read the pdb file, if it exists
    if not pdb_file.filename.endswith(".pdb"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only .pdb files are accepted.")
    if not pdb_file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    if run_output_dirname == "":
        raise HTTPException(status_code=400, detail="The run output directory must be specified.")

    # make the output directory if it doesn't exist
    output_dir = BASE_DIR / run_output_dirname
    raw_output_dir = output_dir / "raw"
    processed_output_dir = output_dir / "processed"
    raw_output_dir.mkdir(parents=True, exist_ok=True)

    # save the uploaded input file
    temp_dir = tempfile.mkdtemp()
    logged_input_pdb = os.path.join(temp_dir, os.path.basename(pdb_file.filename))
    with open(logged_input_pdb, "wb") as f:
        f.write(pdb_file.file.read())
        f.flush()

    # build the command
    cmd = [
        "python", "run.py",
        "--pdb_path", logged_input_pdb,
        "--model_type", model_type,
        "--temperature", str(temperature),
        "--seed", str(random_seed),
        "--redesigned_residues", designed_positions,
        "--batch_size", str(num_seq_per_target),
        "--number_of_batches", "1",
        "--zero_indexed", "1",
        "--out_folder", str(raw_output_dir),
    ]
    if omit_aas is not None and omit_aas != "":
        cmd += ["--omit_AA", omit_aas]
    if ckpt_path is not None:
        cmd += [f"--checkpoint_{model_type}", str(ckpt_path)]
    if plm_bias:
        cmd += ["--plm_bias", plm_bias,
                "--plm_bias_mode", plm_bias_mode,
                "--plm_bias_weight", str(plm_bias_weight)]
    if save_stats:
        cmd += ["--save_stats", "1"]

    start_time = time.time()
    try:
        logging.info(f"Running command: {' '.join(cmd)}")
        _ = sp.run(cmd)
    except sp.CalledProcessError as e:
        logging.error(f"Command failed with error: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: chai-lab a3m-to-pqt failed: {e.stderr}")

    # postprocessing
    seqs_dir = raw_output_dir / "seqs"
    if seqs_dir.exists() and seqs_dir.is_dir():
        shutil.copytree(seqs_dir, processed_output_dir, dirs_exist_ok=True)
    else:
        raise ValueError(f"Expected seqs directory not found at {seqs_dir}")

    # rename all .fa files to .fasta inside processed_output_dir
    for fa_path in processed_output_dir.rglob("*.fa"):
        fasta_path = fa_path.with_suffix(".fasta")
        fa_path.rename(fasta_path)

    # keep only the first chain (chain A, the designed one) in each fasta record
    for fasta_path in processed_output_dir.rglob("*.fasta"):
        processed_records = []
        for record in SeqIO.parse(fasta_path, "fasta"):
            seq_str = str(record.seq).split(":")[0]
            record.seq = record.seq.__class__(seq_str)
            processed_records.append(record)
        with open(fasta_path, "w") as f:
            SeqIO.write(processed_records, f, "fasta")

    elapsed_time = time.time() - start_time

    shutil.rmtree(temp_dir)

    logging.info(f"RCMPNN/{model_type} completed in {elapsed_time:.2f} seconds")


@app.post("/paint/pmpnn")
async def paint_with_pmpnn(
    pdb_file: UploadFile = File(...),
    designed_positions: str = Form(...),
    num_seq_per_target: int = Form(1),
    omit_aas: Optional[str] = Form("C"),
    temperature: Optional[float] = Form(0.1),
    random_seed: Optional[int] = Form(42),
    noise: Optional[str] = Form("0.10"),
    run_output_dirname: str = Form(""),
    plm_bias: Optional[str] = Form(""),
    plm_bias_mode: Optional[str] = Form("aware"),
    plm_bias_weight: Optional[float] = Form(1.0),
    save_stats: bool = Form(False),
):
    ckpt_path = Path("./model_params/proteinmpnn_v_48_020.pt")
    if noise is not None:
        pmpnn_ckpt_file = Path(f"./model_params/proteinmpnn_v_48_{noise.replace('.', '')}.pt")
        if pmpnn_ckpt_file.exists():
            ckpt_path = pmpnn_ckpt_file
        else:
            logging.warning(
                f"Non-existent checkpoint `{pmpnn_ckpt_file}` for RCMPNN/proteinmpnn. Reverting to default: {ckpt_path}."
            )

    await paint_with_an_rcmpnn(
        model_type="protein_mpnn",
        pdb_file=pdb_file,
        designed_positions=designed_positions,
        num_seq_per_target=num_seq_per_target,
        omit_aas=omit_aas,
        temperature=temperature,
        random_seed=random_seed,
        ckpt_path=ckpt_path,
        run_output_dirname=run_output_dirname,
        plm_bias=plm_bias or "",
        plm_bias_mode=plm_bias_mode or "aware",
        plm_bias_weight=plm_bias_weight if plm_bias_weight is not None else 1.0,
        save_stats=save_stats,
    )


@app.post("/paint/hmpnn")
async def paint_with_hmpnn(
    pdb_file: UploadFile = File(...),
    designed_positions: str = Form(...),
    num_seq_per_target: int = Form(1),
    omit_aas: Optional[str] = Form("C"),
    temperature: Optional[float] = Form(0.1),
    random_seed: Optional[int] = Form(42),
    noise: Optional[str] = Form("0.10"),
    run_output_dirname: str = Form(""),
    plm_bias: Optional[str] = Form(""),
    plm_bias_mode: Optional[str] = Form("aware"),
    plm_bias_weight: Optional[float] = Form(1.0),
    save_stats: bool = Form(False),
):
    ckpt_path = Path("./model_params/hypermpnn_v_48_020.pt")
    if noise is not None:
        pmpnn_ckpt_file = Path(f"./model_params/hypermpnn_v_48_{noise.replace('.', '')}.pt")
        if pmpnn_ckpt_file.exists():
            ckpt_path = pmpnn_ckpt_file
        else:
            logging.warning(
                f"Non-existent checkpoint `{pmpnn_ckpt_file}` for RCMPNN/hypermpnn. Reverting to default: {ckpt_path}."
            )

    await paint_with_an_rcmpnn(
        model_type="hyper_mpnn",
        pdb_file=pdb_file,
        designed_positions=designed_positions,
        num_seq_per_target=num_seq_per_target,
        omit_aas=omit_aas,
        temperature=temperature,
        random_seed=random_seed,
        ckpt_path=ckpt_path,
        run_output_dirname=run_output_dirname,
        plm_bias=plm_bias or "",
        plm_bias_mode=plm_bias_mode or "aware",
        plm_bias_weight=plm_bias_weight if plm_bias_weight is not None else 1.0,
        save_stats=save_stats,
    )


@app.post("/paint/lmpnn")
async def paint_with_lmpnn(
    pdb_file: UploadFile = File(...),
    designed_positions: str = Form(...),
    num_seq_per_target: int = Form(1),
    omit_aas: Optional[str] = Form("C"),
    temperature: Optional[float] = Form(0.1),
    random_seed: Optional[int] = Form(42),
    noise: Optional[str] = Form("0.10"),
    run_output_dirname: str = Form(""),
    plm_bias: Optional[str] = Form(""),
    plm_bias_mode: Optional[str] = Form("aware"),
    plm_bias_weight: Optional[float] = Form(1.0),
    save_stats: bool = Form(False),
):
    ckpt_path = Path("./model_params/ligandmpnn_v_32_010_25.pt")
    if noise is not None:
        pmpnn_ckpt_file = Path(f"./model_params/ligandmpnn_v_32_{noise.replace('.', '')}_25.pt")
        if pmpnn_ckpt_file.exists():
            ckpt_path = pmpnn_ckpt_file
        else:
            logging.warning(
                f"Non-existent checkpoint `{pmpnn_ckpt_file}` for RCMPNN/ligandmpnn. Reverting to default: {ckpt_path}."
            )

    await paint_with_an_rcmpnn(
        model_type="ligand_mpnn",
        pdb_file=pdb_file,
        designed_positions=designed_positions,
        num_seq_per_target=num_seq_per_target,
        omit_aas=omit_aas,
        temperature=temperature,
        random_seed=random_seed,
        ckpt_path=ckpt_path,
        run_output_dirname=run_output_dirname,
        plm_bias=plm_bias or "",
        plm_bias_mode=plm_bias_mode or "aware",
        plm_bias_weight=plm_bias_weight if plm_bias_weight is not None else 1.0,
        save_stats=save_stats,
    )


@app.post("/paint/smpnn")
async def paint_with_smpnn(
    pdb_file: UploadFile = File(...),
    designed_positions: str = Form(...),
    num_seq_per_target: int = Form(1),
    omit_aas: Optional[str] = Form("C"),
    temperature: Optional[float] = Form(0.1),
    random_seed: Optional[int] = Form(42),
    noise: Optional[str] = Form("0.10"),
    run_output_dirname: str = Form(""),
    plm_bias: Optional[str] = Form(""),
    plm_bias_mode: Optional[str] = Form("aware"),
    plm_bias_weight: Optional[float] = Form(1.0),
    save_stats: bool = Form(False),
):
    ckpt_path = Path("./model_params/solublempnn_v_48_020.pt")
    if noise is not None:
        pmpnn_ckpt_file = Path(f"./model_params/solublempnn_v_48_{noise.replace('.', '')}.pt")
        if pmpnn_ckpt_file.exists():
            ckpt_path = pmpnn_ckpt_file
        else:
            logging.warning(
                f"Non-existent checkpoint `{pmpnn_ckpt_file}` for RCMPNN/solublempnn. Reverting to default: {ckpt_path}."
            )

    await paint_with_an_rcmpnn(
        model_type="soluble_mpnn",
        pdb_file=pdb_file,
        designed_positions=designed_positions,
        num_seq_per_target=num_seq_per_target,
        omit_aas=omit_aas,
        temperature=temperature,
        random_seed=random_seed,
        ckpt_path=ckpt_path,
        run_output_dirname=run_output_dirname,
        plm_bias=plm_bias or "",
        plm_bias_mode=plm_bias_mode or "aware",
        plm_bias_weight=plm_bias_weight if plm_bias_weight is not None else 1.0,
        save_stats=save_stats,
    )


@app.post("/paint/pmpnn_gml")
async def paint_with_pmpnn_gml(
    pdb_file: UploadFile = File(...),
    designed_positions: str = Form(...),
    num_seq_per_target: int = Form(1),
    omit_aas: Optional[str] = Form("C"),
    temperature: Optional[float] = Form(0.1),
    random_seed: Optional[int] = Form(42),
    run_output_dirname: str = Form(""),
    plm_bias: Optional[str] = Form(""),
    plm_bias_mode: Optional[str] = Form("aware"),
    plm_bias_weight: Optional[float] = Form(1.0),
    save_stats: bool = Form(False),
):
    await paint_with_an_rcmpnn(
        model_type="global_label_membrane_mpnn",
        pdb_file=pdb_file,
        designed_positions=designed_positions,
        num_seq_per_target=num_seq_per_target,
        omit_aas=omit_aas,
        temperature=temperature,
        random_seed=random_seed,
        ckpt_path=None,
        run_output_dirname=run_output_dirname,
        plm_bias=plm_bias or "",
        plm_bias_mode=plm_bias_mode or "aware",
        plm_bias_weight=plm_bias_weight if plm_bias_weight is not None else 1.0,
        save_stats=save_stats,
    )


@app.post("/paint/pmpnn_prml")
async def paint_with_pmpnn_prml(
    pdb_file: UploadFile = File(...),
    designed_positions: str = Form(...),
    num_seq_per_target: int = Form(1),
    omit_aas: Optional[str] = Form("C"),
    temperature: Optional[float] = Form(0.1),
    random_seed: Optional[int] = Form(42),
    run_output_dirname: str = Form(""),
    plm_bias: Optional[str] = Form(""),
    plm_bias_mode: Optional[str] = Form("aware"),
    plm_bias_weight: Optional[float] = Form(1.0),
    save_stats: bool = Form(False),
):
    await paint_with_an_rcmpnn(
        model_type="per_residue_label_membrane_mpnn",
        pdb_file=pdb_file,
        designed_positions=designed_positions,
        num_seq_per_target=num_seq_per_target,
        omit_aas=omit_aas,
        temperature=temperature,
        random_seed=random_seed,
        ckpt_path=None,
        run_output_dirname=run_output_dirname,
        plm_bias=plm_bias or "",
        plm_bias_mode=plm_bias_mode or "aware",
        plm_bias_weight=plm_bias_weight if plm_bias_weight is not None else 1.0,
        save_stats=save_stats,
    )
