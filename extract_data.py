"""Extract SoLEXS (full) and HEL1OS (light curves only) zips into data/."""
import sys
import zipfile
from pathlib import Path

SRC = Path(r"C:\Users\sarth\OneDrive\Pictures\New folder")
DATA = Path(__file__).resolve().parent / "data"

SOLEXS_OUT = DATA / "solexs"
HEL1OS_OUT = DATA / "hel1os"

def safe_member(zip_path, member):
    target = (zip_path.parent / member.filename).resolve()
    root = zip_path.parent.resolve()
    if not str(target).startswith(str(root)):
        raise RuntimeError(f"unsafe path: {member.filename}")
    return target

def extract_file(zip_path, member, dest):
    target = dest / member.filename
    if target.suffix == ".gz" or member.filename.endswith(".gz"):
        target = dest / (member.filename[:-3])
    target.parent.mkdir(parents=True, exist_ok=True)
    with zip_path.open(member) as src, open(target, "wb") as out:
        import shutil
        shutil.copyfileobj(src, out)
    return target

def main():
    slx_zips = sorted(SRC.glob("AL1_SLX_L1_*.zip"))
    hls_zips = sorted(SRC.glob("HLS_*_lev1_V111.zip"))
    print(f"found {len(slx_zips)} SoLEXS zips, {len(hls_zips)} HEL1OS zips")

    for z in slx_zips:
        with zipfile.ZipFile(z) as zf:
            for m in zf.infolist():
                if m.is_dir() or not m.filename.endswith(".gz"):
                    continue
                extract_file(zf, m, SOLEXS_OUT)
    print("SoLEXS done")

    for z in hls_zips:
        with zipfile.ZipFile(z) as zf:
            for m in zf.infolist():
                if m.is_dir():
                    continue
                name = m.filename.split("/")[-1]
                if name.startswith("lightcurve_") and name.endswith(".fits"):
                    extract_file(zf, m, HEL1OS_OUT)
    print("HEL1OS done")

if __name__ == "__main__":
    main()
