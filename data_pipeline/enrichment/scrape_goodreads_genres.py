import argparse
import csv
import json
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.robotparser import RobotFileParser


BASE_DIR = Path(__file__).resolve().parent
SOURCE_FILE = BASE_DIR / "Genre_Teenager_GoodReads_Emotion.csv"
PREVIOUS_RESULT_FILE = BASE_DIR / "Genre_Teenager_GoodReads_Emotion_completed.csv"
OUTPUT_FILE = BASE_DIR / "Genre_Teenager_GoodReads_Emotion_scraped.csv"
CACHE_FILE = BASE_DIR / "goodreads_genre_cache.json"
BASE_URL = "https://www.goodreads.com"
USER_AGENT = "BookGenreResearchBot/1.0"


class GenreParser(HTMLParser):
	def __init__(self):
		super().__init__()
		self.section_depth = None
		self.anchor_depth = None
		self.anchor_text = []
		self.genres = []

	def handle_starttag(self, tag, attrs):
		attrs = dict(attrs)
		if self.section_depth is None:
			if attrs.get("data-testid") == "genresList":
				self.section_depth = 1
			return

		self.section_depth += 1
		if tag == "a" and "/genres/" in (attrs.get("href") or ""):
			self.anchor_depth = self.section_depth
			self.anchor_text = []

	def handle_data(self, data):
		if self.anchor_depth is not None:
			self.anchor_text.append(data)

	def handle_endtag(self, tag):
		if self.section_depth is None:
			return

		if tag == "a" and self.anchor_depth == self.section_depth:
			genre = " ".join("".join(self.anchor_text).split())
			if genre and genre not in self.genres:
				self.genres.append(genre)
			self.anchor_depth = None
			self.anchor_text = []

		self.section_depth -= 1
		if self.section_depth == 0:
			self.section_depth = None


def get_genres(html):
	parser = GenreParser()
	parser.feed(html)
	return parser.genres


def write_rows(rows, fieldnames):
	with OUTPUT_FILE.open("w", encoding="utf-8-sig", newline="") as output:
		writer = csv.DictWriter(output, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def main():
	parser = argparse.ArgumentParser(
		description="Ambil genre publik dari halaman buku Goodreads, dengan jeda dan cache."
	)
	parser.add_argument(
		"--max-pages",
		type=int,
		default=20,
		help="Maksimum halaman baru per proses (default: 20).",
	)
	parser.add_argument(
		"--delay-seconds",
		type=float,
		default=5,
		help="Jeda minimal antarhalaman baru (default: 5 detik).",
	)
	args = parser.parse_args()
	if args.max_pages < 0 or args.delay_seconds < 1:
		parser.error("--max-pages harus >= 0 dan --delay-seconds harus >= 1.")

	input_file = PREVIOUS_RESULT_FILE if PREVIOUS_RESULT_FILE.is_file() else SOURCE_FILE
	if not input_file.is_file():
		raise FileNotFoundError(
			f"File input tidak ditemukan: {SOURCE_FILE.name} atau {PREVIOUS_RESULT_FILE.name}"
		)

	with input_file.open(encoding="utf-8-sig", newline="") as source:
		reader = csv.DictReader(source)
		fieldnames = reader.fieldnames or []
		missing = {"ID", "Genre"}.difference(fieldnames)
		if missing:
			raise ValueError(f"Kolom input tidak ditemukan: {', '.join(sorted(missing))}")
		rows = list(reader)

	if "genre_source" not in fieldnames:
		fieldnames.append("genre_source")

	try:
		cache = json.loads(CACHE_FILE.read_text(encoding="utf-8")) if CACHE_FILE.is_file() else {}
		cache = {
			book_id: result
			for book_id, result in cache.items()
			if isinstance(result, dict) and result.get("genres")
		}
	except (json.JSONDecodeError, OSError):
		print("Peringatan: cache tidak terbaca; cache baru akan dibuat.")
		cache = {}

	robots = RobotFileParser()
	robots.set_url(BASE_URL + "/robots.txt")
	try:
		robots.read()
	except (OSError, URLError) as error:
		raise RuntimeError(f"Tidak dapat memeriksa robots.txt; scraping dihentikan: {error}")

	fetched = 0
	found = 0
	blocked = False
	last_request = 0.0

	try:
		for row in rows:
			if (row.get("Genre") or "").strip():
				continue

			book_id = (row.get("ID") or "").strip()
			if not book_id.isdigit():
				continue

			if book_id in cache:
				genres = cache[book_id].get("genres", [])
				if genres:
					row["Genre"] = ", ".join(genres)
					row["genre_source"] = "goodreads_page"
					found += 1
				continue

			if fetched >= args.max_pages:
				continue

			url = f"{BASE_URL}/book/show/{book_id}"
			if not robots.can_fetch(USER_AGENT, url):
				print("robots.txt tidak mengizinkan halaman buku ini; proses dihentikan.")
				blocked = True
				break

			wait = args.delay_seconds - (time.monotonic() - last_request)
			if last_request and wait > 0:
				time.sleep(wait)
			request = Request(url, headers={"User-Agent": USER_AGENT})
			last_request = time.monotonic()
			fetched += 1
			try:
				with urlopen(request, timeout=30) as response:
					if response.status != 200:
						print(
							f"Goodreads membalas HTTP {response.status} untuk ID {book_id}; "
							"halaman tidak dicache dan proses dihentikan."
						)
						blocked = True
						break
					charset = response.headers.get_content_charset() or "utf-8"
					html = response.read().decode(charset, errors="replace")
			except HTTPError as error:
				if error.code in {403, 429}:
					print(f"Goodreads membatasi akses (HTTP {error.code}); proses dihentikan.")
					blocked = True
					break
				print(f"Gagal mengambil ID {book_id}: HTTP {error.code}; ID dilewati.")
				continue
			except (URLError, TimeoutError) as error:
				print(f"Gagal mengambil ID {book_id}: {error}; proses dihentikan.")
				blocked = True
				break

			genres = get_genres(html)
			if genres:
				cache[book_id] = {"genres": genres}
				CACHE_FILE.write_text(
					json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8"
				)
				row["Genre"] = ", ".join(genres)
				row["genre_source"] = "goodreads_page"
				found += 1
			else:
				print(f"ID {book_id}: genre tidak ditemukan; hasil tidak dicache.")
			print(f"ID {book_id}: {len(genres)} genre ditemukan")
			write_rows(rows, fieldnames)
	except KeyboardInterrupt:
		print("Dihentikan oleh pengguna; hasil parsial sedang disimpan.")
	finally:
		write_rows(rows, fieldnames)

	print(f"File hasil: {OUTPUT_FILE.name}")
	print(f"File sumber: {input_file.name}")
	print(f"Halaman baru diambil: {fetched}")
	print(f"Baris genre terisi dari cache/halaman: {found}")
	print(f"ID tersimpan di cache: {len(cache)}")
	print(f"Request dibatasi/dihentikan: {blocked}")
	print("Pastikan penggunaan data dan otomasi sesuai ketentuan Goodreads.")


if __name__ == "__main__":
	main()