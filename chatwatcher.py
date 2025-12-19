import time
from typing import Any, Iterator

import pytchat

def watch_chat(video_id: str) -> Iterator[Any]:
    chat = pytchat.create(video_id=video_id)
    while chat.is_alive():
        for c in chat.get().sync_items():
            yield c

def watch_loop(video_id: str) -> Iterator[Any]:
    while True:
        try:
            print("Creating new pytchat session...")
            yield from watch_chat(video_id)
            print("Chat session is no longer alive, restarting in 2 seconds...")
            time.sleep(2)
        except Exception as e:
            print(f"Exception in chat_watcher: {e}. Restarting in 10 seconds...")
            time.sleep(10)
