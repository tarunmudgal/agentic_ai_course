import os

from dotenv import load_dotenv


def main() -> None:
    load_dotenv()
    db_url = os.getenv("PGVECTOR_DATABASE_URL")

    print("Day 5 pgvector demo")
    print("Use this SQL to initialize pgvector:")
    print("CREATE EXTENSION IF NOT EXISTS vector;")
    print("CREATE TABLE IF NOT EXISTS notes (id serial PRIMARY KEY, topic text, embedding vector(384));")
    print("CREATE INDEX IF NOT EXISTS notes_embedding_idx ON notes USING ivfflat (embedding vector_cosine_ops);")

    if db_url:
        print("PGVECTOR_DATABASE_URL found. You can now connect with psycopg2 in a follow-up exercise.")
    else:
        print("Set PGVECTOR_DATABASE_URL in .env to run live DB operations.")


if __name__ == "__main__":
    main()

