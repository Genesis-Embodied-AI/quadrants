import sys
import time
from pathlib import Path

import requests
import tqdm


def download_file(url, destination, name):
    """Retry transient failures, including interruptions while streaming the response body."""
    destination = Path(destination)
    partial = destination.with_name(destination.name + ".download")
    attempts = 5
    for attempt in range(1, attempts + 1):
        try:
            with requests.get(url, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                with (
                    tqdm.tqdm(
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        total=int(response.headers.get("content-length", 0)),
                        desc=name,
                    ) as progress,
                    partial.open("wb") as output,
                ):
                    for chunk in response.iter_content(chunk_size=8192):
                        progress.update(output.write(chunk))
            partial.replace(destination)
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
            partial.unlink(missing_ok=True)
