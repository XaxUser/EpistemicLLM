from dotenv import load_dotenv
import os

load_dotenv()

key = os.environ.get("OPENAI_API_KEY")

if not key:
    print("❌ Key not found. Check your .env file.")
elif not key.startswith("sk-"):
    print(f"❌ Key found but looks wrong: {key[:10]}...")
else:
    print(f"✓ Key loaded correctly: {key[:8]}...{key[-4:]}")
    print(f"✓ Key length: {len(key)} characters")