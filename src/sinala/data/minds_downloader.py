import hashlib
import stat
import subprocess
import zipfile
from pathlib import Path
from typing import Final

import httpx
from loguru import logger
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

SIGNER_ARCHIVES: Final[dict[str, dict[str, str]]] = {
    "01": {"filename": "Sinalizador01.zip", "md5": "0259b309d2d5472f20d9842547d94a8c"},
    "02": {"filename": "Sinalizador02.zip", "md5": "e2b8673264ba7fca4cfc071654a099e8"},
    "03": {"filename": "Sinalizador03.zip", "md5": "18996a6bdc404e98a854bfd8a39d0268"},
    "04": {"filename": "Sinalizador04.zip", "md5": "f244c4b3fbc99854b84b3e1c166c9361"},
    "05": {"filename": "Sinalizador05.zip", "md5": "0382984d3bd6b04d1120aebc3e6d637b"},
    "06": {"filename": "Sinalizador06.zip", "md5": "d530e828ab775046f9e1d0938525ab3f"},
    "07": {"filename": "Sinalizador07.zip", "md5": "8e885f5cb6823744a583f206e4d2051c"},
    "08": {"filename": "Sinalizador08.zip", "md5": "635bd029bb1fd59a2a787b8b35e7645a"},
    "09": {"filename": "Sinalizador09.zip", "md5": "07012d697449efccf53783fcc4394a95"},
    "10": {"filename": "Sinalizador10.zip", "md5": "d6012f3e49e95d0cef7f6ec888743baa"},
    "11": {"filename": "Sinalizador11.zip", "md5": "a4653a029f7d4ea90ca4745d8390e133"},
    "12": {"filename": "Sinalizador12.zip", "md5": "fff0716f21025cf8a3739c7e61d70db8"},
}
_ZENODO_FILE_URL = "https://zenodo.org/records/2667329/files/{filename}?download=1"


class MindsDatasetDownloader:
    def __init__(self, output_dir: Path, timeout_seconds: float = 120.0) -> None:
        self.output_dir = output_dir
        self.timeout_seconds = timeout_seconds

    def download_signers(self, signer_ids: list[str]) -> list[Path]:
        unknown_signers = sorted(set(signer_ids) - set(SIGNER_ARCHIVES))
        if unknown_signers:
            raise ValueError(f"Unknown signer ids: {', '.join(unknown_signers)}")

        self.output_dir.mkdir(parents=True, exist_ok=True)
        downloaded_paths: list[Path] = []
        with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True) as client:
            for signer_id in signer_ids:
                metadata = SIGNER_ARCHIVES[signer_id]
                destination = self.output_dir / metadata["filename"]
                if destination.exists() and self._md5(destination) == metadata["md5"]:
                    downloaded_paths.append(destination)
                    continue
                self._download_with_retry(client, signer_id, destination, metadata["md5"])
                downloaded_paths.append(destination)
        return downloaded_paths

    @retry(
        retry=retry_if_exception_type((httpx.HTTPError, OSError, ValueError)),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _download_with_retry(self, client: httpx.Client, signer_id: str, destination: Path, expected_md5: str) -> None:
        filename = SIGNER_ARCHIVES[signer_id]["filename"]
        url = _ZENODO_FILE_URL.format(filename=filename)
        temporary_path = destination.with_suffix(destination.suffix + ".part")
        offset = temporary_path.stat().st_size if temporary_path.exists() else 0
        headers = {"Range": f"bytes={offset}-"} if offset else {}
        logger.info("Downloading {} from byte {}", filename, offset)
        with client.stream("GET", url, headers=headers) as response:
            response.raise_for_status()
            append = offset > 0 and response.status_code == httpx.codes.PARTIAL_CONTENT
            mode = "ab" if append else "wb"
            with temporary_path.open(mode) as output:
                for chunk in response.iter_bytes(chunk_size=1024 * 1024):
                    output.write(chunk)

        if self._md5(temporary_path) != expected_md5:
            if not append:
                temporary_path.unlink(missing_ok=True)
            raise ValueError(f"MD5 mismatch for {filename}")
        temporary_path.replace(destination)

    def extract_signers(self, signer_ids: list[str], output_dir: Path) -> list[Path]:
        unknown_signers = sorted(set(signer_ids) - set(SIGNER_ARCHIVES))
        if unknown_signers:
            raise ValueError(f"Unknown signer ids: {', '.join(unknown_signers)}")
        if output_dir.is_symlink():
            raise ValueError(f"Extraction directory cannot be a symlink: {output_dir}")
        output_dir.mkdir(parents=True, exist_ok=True)
        extraction_root = output_dir.resolve()
        extracted_paths: list[Path] = []
        for signer_id in signer_ids:
            metadata = SIGNER_ARCHIVES[signer_id]
            archive_path = self.output_dir / metadata["filename"]
            if not archive_path.exists() or self._md5(archive_path) != metadata["md5"]:
                raise ValueError(f"Archive is missing or invalid: {archive_path}")
            with zipfile.ZipFile(archive_path) as archive:
                for member in archive.infolist():
                    destination = (extraction_root / member.filename).resolve()
                    if not destination.is_relative_to(extraction_root):
                        raise ValueError(f"Unsafe archive member: {member.filename}")
                    mode = (member.external_attr >> 16) & 0o170000
                    if stat.S_ISLNK(mode):
                        raise ValueError(f"Symlink archive member is not allowed: {member.filename}")
            subprocess.run(["/usr/bin/ditto", "-x", "-k", str(archive_path), str(output_dir)], check=True)
            extracted_paths.append(output_dir / f"Sinalizador{signer_id}")
        return extracted_paths

    @staticmethod
    def _md5(path: Path) -> str:
        digest = hashlib.md5()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


__all__ = ["MindsDatasetDownloader", "SIGNER_ARCHIVES"]
