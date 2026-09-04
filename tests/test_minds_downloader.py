import hashlib
from pathlib import Path

from sinala.data.minds_downloader import MindsDatasetDownloader


def test_minds_downloader_when_file_is_written_then_calculates_md5(tmp_path: Path) -> None:
    archive = tmp_path / "archive.zip"
    content = b"sinala"
    archive.write_bytes(content)
    downloader = MindsDatasetDownloader(tmp_path)

    checksum = downloader._md5(archive)

    assert checksum == hashlib.md5(content).hexdigest()
