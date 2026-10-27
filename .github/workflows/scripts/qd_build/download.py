import sys
import time
from pathlib import Path

import requests
import tqdm


def download_file(url, destination, name):
    """Retry transient failures, including interruptions while streaming the response body."""
    destination = Path(destination)
    partial_file = destination.with_name(destination.name + ".download")
    attempts = 5
    for attempt in range(1, attempts + 1):
        try:
            # Allow 15 seconds to connect and 60 seconds without receiving data.
            with requests.get(url, stream=True, timeout=(15, 60)) as response:
                # Raise HTTPError if the GET request above returned a 4xx or 5xx status.
                response.raise_for_status()
                with (
                    tqdm.tqdm(
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        total=int(response.headers.get("content-length", 0)),
                        desc=name,
                    ) as progress,
                    partial_file.open("wb") as output_fh,
                ):
                    for chunk in response.iter_content(chunk_size=8192):
                        progress.update(output_fh.write(chunk))
            partial_file.replace(destination)
            return
        except (
            requests.ConnectionError,
            requests.Timeout,
            requests.exceptions.ChunkedEncodingError,
            requests.HTTPError,
        ) as error:
            # Raising aborts retries.
            if isinstance(error, requests.HTTPError):
                # Only retry on these specific HTTP status codes.
                # 408: Request Timeout; 429: Too Many Requests; 500: Internal Server Error;
                # 502: Bad Gateway; 503: Service Unavailable; 504: Gateway Timeout.
                if error.response.status_code not in (408, 429, 500, 502, 503, 504):
                    raise
            if attempt >= attempts:
                raise
            # Delay: 1, 2, 4, 8, ... seconds.
            delay = 2 ** (attempt - 1)
            print(
                f"Download {name} failed (attempt {attempt}/{attempts}): {error}. Retrying in {delay}s.",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
        finally:
            # Don't leave partial downloads around
            partial_file.unlink(missing_ok=True)
