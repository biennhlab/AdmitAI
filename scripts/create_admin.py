import asyncio
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from src.database.models import User, Base
from src.auth.security import hash_password
from src.config import settings

async def create_admin(username, password, name):
    engine = create_async_engine(settings.DATABASE_URL)
    async_session = async_sessionmaker(engine, expire_on_commit=False)
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    async with async_session() as session:
        user = User(
            username=username,
            password_hash=hash_password(password),
            full_name=name,
            role="admin"
        )
        session.add(user)
        await session.commit()
        print(f"Admin user '{username}' created successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create admin user")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    
    asyncio.run(create_admin(args.username, args.password, args.name))
