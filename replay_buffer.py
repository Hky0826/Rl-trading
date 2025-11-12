# File: replay_buffer.py
# Description: A persistent replay buffer to store trading experiences for offline analysis or training.
# =============================================================================
import os
import logging
import joblib
from collections import deque
import config

class ReplayBuffer:
    """
    A persistent replay buffer that saves experiences to disk using joblib.
    It loads existing experiences upon initialization, allowing the buffer to
    grow across multiple sessions of the live trading script.
    """
    def __init__(self):
        """
        Initializes the ReplayBuffer. It defines the file path from the config
        and loads any previously saved experiences.
        """
        self.file_path = os.path.join(config.TRADE_LOG_DIR, config.REPLAY_BUFFER_FILE)
        self.buffer = deque()
        self.load()

    def add(self, experience):
        """
        Adds a single experience to the buffer.

        Args:
            experience (tuple): A tuple representing the state, action, reward,
                                next_state, and done flag.
        """
        self.buffer.append(experience)

    def save(self):
        """
        Saves the entire buffer to a file using joblib.
        It ensures the directory exists before saving.
        """
        try:
            # Ensure the directory for the replay buffer exists
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            joblib.dump(self.buffer, self.file_path)
            logging.info(f"Replay buffer with {len(self.buffer)} experiences saved to {self.file_path}")
        except Exception as e:
            logging.error(f"Error saving replay buffer: {e}")

    def load(self):
        """
        Loads the buffer from a file if it exists.
        Handles potential errors if the file is corrupted or not found.
        """
        if os.path.exists(self.file_path):
            try:
                self.buffer = joblib.load(self.file_path)
                logging.info(f"Successfully loaded replay buffer with {len(self.buffer)} experiences from {self.file_path}")
            except Exception as e:
                logging.error(f"Could not load replay buffer from {self.file_path}. Starting fresh. Error: {e}")
                self.buffer = deque()
        else:
            logging.info("No existing replay buffer found. Starting a new one.")

    def __len__(self):
        """Returns the current number of experiences in the buffer."""
        return len(self.buffer)