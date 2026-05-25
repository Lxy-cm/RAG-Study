from qa_storage import DB_PATH, migrate_json_store


if __name__ == "__main__":
    count = migrate_json_store()
    print(f"已迁移 {count} 条历史消息到 SQLite: {DB_PATH}")
