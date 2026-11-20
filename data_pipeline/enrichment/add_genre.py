import argparse
import csv
import json
import os
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


BASE_DIR = Path(__file__).resolve().parent
GOODREADS_FILE = BASE_DIR / "GoodReads_100k_books.csv"
TEENAGER_FILE = BASE_DIR / "Genre_Teenager_GoodReads_Emotion.csv"
OUTPUT_FILE = BASE_DIR / "Genre_Teenager_GoodReads_Emotion_completed.csv"
API_CACHE_FILE = BASE_DIR / "google_books_cache.json"
GOOGLE_BOOKS_API = "https://www.googleapis.com/books/v1/volumes"


def normalize_isbn(value):
	return re.sub(r"[^0-9Xx]", "", str(value or "")).upper()


def normalize_text(value):
	text = unicodedata.normalize("NFKD", str(value or "").casefold())
	text = text.encode("ascii", "ignore").decode("ascii")
	return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", text)).strip()


def isbn10_cores(value):
	isbn = normalize_isbn(value)
	if len(isbn) == 10 and isbn[:9].isdigit():
		return {isbn[:9]}
	if len(isbn) == 9 and isbn.isdigit():
		# A 9-digit value may have lost its check digit or a leading zero.
		return {isbn, "0" + isbn[:8]}
	return set()


def teenager_cores(value):
	isbn = normalize_isbn(value)
	if len(isbn) == 13 and isbn.isdigit() and isbn.startswith("978"):
		return {isbn[3:12]}
	return isbn10_cores(isbn)


def author_matches(expected, authors):
	expected = normalize_text(expected)
	if not expected:
		return True
	return any(
		expected in normalize_text(author) or normalize_text(author) in expected
		for author in authors
		if normalize_text(author)
	)


def isbn_matches(expected, identifiers):
	expected = normalize_isbn(expected)
	return bool(expected) and any(
		normalize_isbn(identifier.get("identifier")) == expected
		for identifier in identifiers
	)


def google_books_genre(row, search, by_isbn):
	title = normalize_text(row.get("Title"))
	author = row.get("Author") or ""
	accepted = []
	for item in search:
		info = item.get("volumeInfo", {})
		categories = info.get("categories", [])
		if not categories:
			continue

		authors = info.get("authors", [])
		identifiers = info.get("industryIdentifiers", [])
		identifier_match = isbn_matches(row.get("ISBN"), identifiers)
		result_title = normalize_text(info.get("title"))
		title_match = (
			result_title == title
			or (title and result_title and (title in result_title or result_title in title))
		)
		if (by_isbn and identifier_match) or (title_match and author_matches(author, authors)):
			accepted.append(tuple(category.strip() for category in categories if category.strip()))

	unique = {categories for categories in accepted if categories}
	if len(unique) == 1:
		return ", ".join(next(iter(unique)))
	return ""


def main():
	parser = argparse.ArgumentParser(description="Tambahkan genre buku dari Goodreads dan Google Books.")
	parser.add_argument(
		"--api",
		action="store_true",
		help="Cari genre yang belum ditemukan melalui Google Books API.",
	)
	parser.add_argument(
		"--max-api-requests",
		type=int,
		default=100,
		help="Batas request Google Books API per proses (default: 100).",
	)
	args = parser.parse_args()
	if args.max_api_requests < 0:
		parser.error("--max-api-requests harus bernilai 0 atau lebih.")

	if not GOODREADS_FILE.is_file() or not TEENAGER_FILE.is_file():
		raise FileNotFoundError(
			"Pastikan GoodReads_100k_books.csv dan "
			"Teenager_GoodReads_Emotion.csv berada di folder yang sama "
			"dengan add_genre.py."
		)

	genres_by_core = defaultdict(set)
	books_by_title = defaultdict(list)
	with GOODREADS_FILE.open(encoding="utf-8-sig", newline="") as source:
		reader = csv.DictReader(source)
		required = {"isbn", "isbn13", "genre", "title", "author"}
		missing = required.difference(reader.fieldnames or [])
		if missing:
			raise ValueError(f"Kolom Goodreads tidak ditemukan: {', '.join(sorted(missing))}")

		for row in reader:
			genre = (row.get("genre") or "").strip()
			cores = isbn10_cores(row.get("isbn"))
			isbn13 = normalize_isbn(row.get("isbn13"))
			if len(isbn13) == 13 and isbn13.isdigit():
				if isbn13.startswith("978"):
					cores.add(isbn13[3:12])
				if genre:
					genres_by_core["isbn13:" + isbn13].add(genre)

			if genre:
				for core in cores:
					genres_by_core["core:" + core].add(genre)

			title = normalize_text(row.get("title"))
			if title and genre:
				books_by_title[title].append(
					{"author": normalize_text(row.get("author")), "genre": genre}
				)

	with TEENAGER_FILE.open(encoding="utf-8-sig", newline="") as source:
		reader = csv.DictReader(source)
		fieldnames = reader.fieldnames or []
		missing = {"ISBN", "Title", "Author"}.difference(fieldnames)
		if missing:
			raise ValueError(f"Kolom Teenager tidak ditemukan: {', '.join(sorted(missing))}")
		rows = list(reader)

	genre_field = next((name for name in fieldnames if name.casefold() == "genre"), None)
	if genre_field is None:
		genre_field = "Genre"
		fieldnames.append(genre_field)
	if "genre_source" not in fieldnames:
		fieldnames.append("genre_source")

	already_filled_rows = sum(bool((row.get(genre_field) or "").strip()) for row in rows)
	isbn_filled_rows = 0
	title_filled_rows = 0
	api_filled_rows = 0
	ambiguous_rows = 0
	api_requests = 0
	api_disabled = False
	last_api_request = 0.0
	api_cache = {}
	if API_CACHE_FILE.is_file():
		try:
			loaded_cache = json.loads(API_CACHE_FILE.read_text(encoding="utf-8"))
			api_cache = {
				query: result
				for query, result in loaded_cache.items()
				if isinstance(result, dict) or bool(result)
			}
		except (json.JSONDecodeError, OSError):
			print("Peringatan: cache Google Books tidak terbaca; cache baru akan dibuat.")

	def search_google_books(query):
		nonlocal api_requests, api_disabled, last_api_request
		query = (query or "").strip()
		if not query:
			return None
		if query in api_cache:
			cached = api_cache[query]
			return cached["items"] if isinstance(cached, dict) else cached
		if api_disabled or api_requests >= args.max_api_requests:
			return None

		params = {"q": query, "maxResults": 10}
		api_key = os.environ.get("GOOGLE_BOOKS_API_KEY")
		if api_key:
			params["key"] = api_key
		url = GOOGLE_BOOKS_API + "?" + urlencode(params)
		request = Request(url, headers={"User-Agent": "BookGenreMatcher/1.0"})
		for attempt in range(4):
			wait = 1.2 - (time.monotonic() - last_api_request)
			if wait > 0:
				time.sleep(wait)
			last_api_request = time.monotonic()
			api_requests += 1
			try:
				with urlopen(request, timeout=20) as response:
					payload = json.loads(response.read().decode("utf-8"))
				results = payload.get("items", [])
				break
			except HTTPError as error:
				if error.code == 429:
					if attempt < 3:
						retry_after = error.headers.get("Retry-After", "")
						try:
							delay = max(float(retry_after), 2 ** attempt)
						except ValueError:
							delay = 2 ** attempt
						print(f"Google Books membatasi request; mencoba lagi dalam {delay:g} detik.")
						time.sleep(delay)
						continue
					api_disabled = True
					print("Peringatan: API tetap memberi HTTP 429; pencarian API dihentikan untuk proses ini.")
				else:
					print(f"Peringatan: Google Books API gagal untuk query {query!r}: {error}")
				return None
			except (URLError, TimeoutError, json.JSONDecodeError) as error:
				print(f"Peringatan: Google Books API gagal untuk query {query!r}: {error}")
				return None
		else:
			return None

		api_cache[query] = {"items": results}
		API_CACHE_FILE.write_text(
			json.dumps(api_cache, ensure_ascii=False, indent=2), encoding="utf-8"
		)
		return results

	for row in rows:
		if (row.get(genre_field) or "").strip():
			continue

		isbn = normalize_isbn(row.get("ISBN"))
		keys = {"core:" + core for core in teenager_cores(isbn)}
		if len(isbn) == 13 and isbn.isdigit():
			keys.add("isbn13:" + isbn)

		matches = set()
		for key in keys:
			matches.update(genres_by_core.get(key, set()))

		if len(matches) == 1:
			row[genre_field] = matches.pop()
			row["genre_source"] = "Goodreads ISBN"
			isbn_filled_rows += 1
			continue

		title = normalize_text(row.get("Title"))
		candidates = books_by_title.get(title, [])
		candidate_genres = {book["genre"] for book in candidates}
		if len(candidate_genres) > 1:
			author = normalize_text(row.get("Author"))
			if author:
				by_author = [
					book for book in candidates
					if book["author"]
					and (author == book["author"] or author in book["author"] or book["author"] in author)
				]
				if by_author:
					candidates = by_author
					candidate_genres = {book["genre"] for book in candidates}

		if len(candidate_genres) == 1:
			row[genre_field] = candidate_genres.pop()
			row["genre_source"] = "Goodreads title"
			title_filled_rows += 1
			continue

		if args.api:
			genre = ""
			isbn = normalize_isbn(row.get("ISBN"))
			if len(isbn) in {10, 13}:
				results = search_google_books("isbn:" + isbn)
				if results is not None:
					genre = google_books_genre(row, results, by_isbn=True)

			title_query = (row.get("Title") or "").strip()
			if not genre and title_query:
				author = (row.get("Author") or "").strip()
				if author:
					title_query += " " + author
				results = search_google_books(title_query)
				if results is not None:
					genre = google_books_genre(row, results, by_isbn=False)

			if not genre and (row.get("Title") or "").strip():
				results = search_google_books((row.get("Title") or "").strip())
				if results is not None:
					genre = google_books_genre(row, results, by_isbn=False)

			if genre:
				row[genre_field] = genre
				row["genre_source"] = "Google Books API"
				api_filled_rows += 1
				continue

		if matches or candidates:
			ambiguous_rows += 1
		row["genre_source"] = "" 

	with OUTPUT_FILE.open("w", encoding="utf-8-sig", newline="") as output:
		writer = csv.DictWriter(output, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

	print(f"File hasil: {OUTPUT_FILE.name}")
	print(f"Jumlah baris Teenager: {len(rows)}")
	newly_filled_rows = isbn_filled_rows + title_filled_rows + api_filled_rows
	filled_rows = already_filled_rows + newly_filled_rows
	print(f"Genre yang sudah terisi dipertahankan: {already_filled_rows}")
	print(f"Genre cocok melalui ISBN: {isbn_filled_rows}")
	print(f"Tambahan genre melalui judul: {title_filled_rows}")
	print(f"Tambahan genre melalui Google Books API: {api_filled_rows}")
	if args.api:
		print(f"Request API baru: {api_requests} (cache: {len(api_cache)})")
		if api_disabled:
			print("Pencarian API dihentikan karena batas request; jalankan lagi nanti atau gunakan API key.")
	print(f"Genre berhasil ditambahkan total: {filled_rows}")
	print(f"ISBN dengan genre berbeda/ambigu: {ambiguous_rows}")
	print(f"Baris tanpa genre: {len(rows) - filled_rows}")
	print("Judul duplikat yang tidak dapat dipastikan dibiarkan tanpa genre.")
	if api_filled_rows:
		print("Catatan: kategori Google Books mungkin berbeda dari genre Goodreads.")


if __name__ == "__main__":
	main()
