# File: performance_logger.py
# Description: (ENHANCED) Robust performance logging with background IO and lock timeouts
# =============================================================================
import json
import os
import logging
from datetime import datetime
import threading
import config

class PerformanceLogger:
    def __init__(self):
        self.file_path = os.path.join(config.TRADE_LOG_DIR, "live_performance_log.json")
        self.backup_path = os.path.join(config.TRADE_LOG_DIR, "live_performance_log_backup.json")
        self.lock = threading.Lock()  # Thread safety for file operations

        # Ensure directory exists
        os.makedirs(config.TRADE_LOG_DIR, exist_ok=True)

        self.stats = self._load_stats()
        self.session_processed_deals = set(self.stats.get('processed_deals', []))

        # Auto-save counter
        self.trades_since_save = 0
        self.equity_updates_since_save = 0

        # Background save thread state
        self._save_thread_lock = threading.Lock()
        self._last_emergency_attempt = None

        # Ensure a file exists immediately
        if not os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'w') as f:
                    json.dump(self.stats, f, indent=4, default=str)
                logging.info("Created new performance log file at startup")
            except Exception as e:
                logging.error(f"Failed to create initial performance log: {e}")

    # -------------------------
    # Loading & validation
    # -------------------------
    def _load_stats(self):
        """Loads existing stats from the log file with fallback to backup."""
        stats = self._try_load_file(self.file_path)
        if stats:
            return stats

        logging.warning("Main performance log corrupted or missing, trying backup...")
        stats = self._try_load_file(self.backup_path)
        if stats:
            # Restore from backup (best-effort, don't crash if cannot write)
            try:
                with open(self.file_path, 'w') as f:
                    json.dump(stats, f, indent=4, default=str)
                logging.info("Restored performance log from backup")
            except Exception as e:
                logging.error(f"Could not restore from backup: {e}")
            return stats

        logging.info("Creating new performance log")
        return self._get_initial_stats()

    def _try_load_file(self, filepath):
        """Attempts to load stats from a specific file."""
        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, 'r') as f:
                content = f.read()
                if not content.strip():
                    return None
                stats = json.loads(content)
                stats = self._validate_and_repair_stats(stats)
                return stats
        except (json.JSONDecodeError, IOError, ValueError) as e:
            logging.error(f"Error loading {filepath}: {e}")
            return None

    def _validate_and_repair_stats(self, stats):
        """Validates stats structure and repairs missing fields."""
        template = self._get_initial_stats()

        for key in template:
            if key not in stats:
                logging.warning(f"Missing key '{key}' in stats, adding default value")
                stats[key] = template[key]

        # Ensure trades_by_risk_level has all risk levels
        for level in config.RISK_LEVELS:
            level_key = str(level)
            if level_key not in stats['trades_by_risk_level']:
                stats['trades_by_risk_level'][level_key] = {'trades': 0, 'wins': 0}
            elif not isinstance(stats['trades_by_risk_level'][level_key], dict):
                stats['trades_by_risk_level'][level_key] = {'trades': 0, 'wins': 0}

        # Ensure trades_by_rr_profile has all profiles
        for i in range(len(config.RR_PROFILES)):
            rr_key = str(i)
            if rr_key not in stats['trades_by_rr_profile']:
                stats['trades_by_rr_profile'][rr_key] = {'trades': 0, 'wins': 0}
            elif not isinstance(stats['trades_by_rr_profile'][rr_key], dict):
                stats['trades_by_rr_profile'][rr_key] = {'trades': 0, 'wins': 0}

        # Ensure processed_deals is a list
        if not isinstance(stats.get('processed_deals'), list):
            stats['processed_deals'] = []

        # Ensure equity_history is a list
        if not isinstance(stats.get('equity_history'), list):
            stats['equity_history'] = [{"time": datetime.now().isoformat(), "equity": config.INITIAL_EQUITY}]

        # Convert numeric strings to proper types
        for key in ['total_trades', 'wins']:
            if isinstance(stats.get(key), str):
                try:
                    stats[key] = int(stats[key])
                except ValueError:
                    stats[key] = 0

        for key in ['gross_profit', 'gross_loss']:
            if isinstance(stats.get(key), str):
                try:
                    stats[key] = float(stats[key])
                except ValueError:
                    stats[key] = 0.0

        return stats

    def _get_initial_stats(self):
        trades_by_risk = {str(level): {'trades': 0, 'wins': 0} for level in config.RISK_LEVELS}
        trades_by_rr = {str(i): {'trades': 0, 'wins': 0} for i in range(len(config.RR_PROFILES))}

        return {
            "total_trades": 0,
            "wins": 0,
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "trades_by_risk_level": trades_by_risk,
            "trades_by_rr_profile": trades_by_rr,
            "equity_history": [{"time": datetime.now().isoformat(), "equity": config.INITIAL_EQUITY}],
            "processed_deals": [],
            "last_save_time": datetime.now().isoformat()
        }

    # -------------------------
    # Logging methods
    # -------------------------
    def log_trade(self, deal, risk_level, rr_profile_index):
        """Logs a single completed trade with thread safety."""
        if not deal:
            logging.warning("Attempted to log null deal")
            return

        # Fast path: try to acquire lock quickly, otherwise skip logging save (still update in-memory)
        acquired = self.lock.acquire(timeout=0.5)
        try:
            # When lock acquired, perform updates
            if acquired:
                try:
                    if deal.ticket in self.session_processed_deals:
                        return

                    self.stats['total_trades'] += 1
                    pnl = float(deal.profit) if hasattr(deal, 'profit') else 0.0

                    if pnl > 0:
                        self.stats['wins'] += 1
                        self.stats['gross_profit'] += pnl
                    else:
                        self.stats['gross_loss'] += abs(pnl)

                    risk_key = str(float(risk_level))
                    if risk_key in self.stats['trades_by_risk_level']:
                        self.stats['trades_by_risk_level'][risk_key]['trades'] += 1
                        if pnl > 0:
                            self.stats['trades_by_risk_level'][risk_key]['wins'] += 1
                    else:
                        logging.warning(f"Unknown risk level: {risk_key}")

                    rr_key = str(int(rr_profile_index))
                    if rr_key in self.stats['trades_by_rr_profile']:
                        self.stats['trades_by_rr_profile'][rr_key]['trades'] += 1
                        if pnl > 0:
                            self.stats['trades_by_rr_profile'][rr_key]['wins'] += 1
                    else:
                        logging.warning(f"Unknown RR profile index: {rr_key}")

                    # Mark as processed
                    self.stats['processed_deals'].append(deal.ticket)
                    self.session_processed_deals.add(deal.ticket)

                    self.trades_since_save += 1

                    # Auto-save after every N trades: spawn background save to avoid blocking
                    if self.trades_since_save >= 5:
                        self.trades_since_save = 0
                        self._spawn_background_save()
                except Exception as e:
                    logging.error(f"Error logging trade (while holding lock): {e}")
            else:
                # Could not acquire lock quickly: perform in-memory updates without saving
                try:
                    if deal.ticket in self.session_processed_deals:
                        return

                    self.stats['total_trades'] += 1
                    pnl = float(deal.profit) if hasattr(deal, 'profit') else 0.0

                    if pnl > 0:
                        self.stats['wins'] += 1
                        self.stats['gross_profit'] += pnl
                    else:
                        self.stats['gross_loss'] += abs(pnl)

                    risk_key = str(float(risk_level))
                    if risk_key in self.stats['trades_by_risk_level']:
                        self.stats['trades_by_risk_level'][risk_key]['trades'] += 1
                        if pnl > 0:
                            self.stats['trades_by_risk_level'][risk_key]['wins'] += 1

                    rr_key = str(int(rr_profile_index))
                    if rr_key in self.stats['trades_by_rr_profile']:
                        self.stats['trades_by_rr_profile'][rr_key]['trades'] += 1
                        if pnl > 0:
                            self.stats['trades_by_rr_profile'][rr_key]['wins'] += 1

                    self.stats['processed_deals'].append(deal.ticket)
                    self.session_processed_deals.add(deal.ticket)
                except Exception as e:
                    logging.error(f"Error logging trade (lock busy): {e}")
        finally:
            if acquired:
                try:
                    self.lock.release()
                except Exception:
                    pass

    def log_equity(self, current_equity):
        """Logs current equity with pruning of old data. Uses background saves when thresholds reached."""
        try:
            current_equity = float(current_equity)
        except Exception:
            logging.warning(f"Invalid equity value passed to log_equity: {current_equity}")
            return

        acquired = self.lock.acquire(timeout=0.5)
        try:
            if acquired:
                try:
                    self.stats['equity_history'].append({
                        "time": datetime.now().isoformat(),
                        "equity": current_equity
                    })

                    # Prune old equity history (keep last 10,000 points)
                    max_equity_points = 10000
                    if len(self.stats['equity_history']) > max_equity_points:
                        self.stats['equity_history'] = self.stats['equity_history'][-max_equity_points:]

                    self.equity_updates_since_save += 1

                    # Auto-save after every 100 equity updates
                    if self.equity_updates_since_save >= 100:
                        self.equity_updates_since_save = 0
                        self._spawn_background_save()
                except Exception as e:
                    logging.error(f"Error logging equity (while holding lock): {e}")
            else:
                # Lock busy: append to in-memory history without saving
                try:
                    self.stats['equity_history'].append({
                        "time": datetime.now().isoformat(),
                        "equity": current_equity
                    })
                    # ensure we still prune occasionally
                    if len(self.stats['equity_history']) > 15000:
                        self.stats['equity_history'] = self.stats['equity_history'][-10000:]
                except Exception as e:
                    logging.error(f"Error logging equity (lock busy): {e}")
        finally:
            if acquired:
                try:
                    self.lock.release()
                except Exception:
                    pass

    # -------------------------
    # Saving (background)
    # -------------------------
    def _spawn_background_save(self):
        """Start a background thread to perform save_stats if none is currently running."""
        # Prevent too many concurrent save threads
        if not self._save_thread_lock.acquire(blocking=False):
            # another save thread is active
            return

        def _worker():
            try:
                # Call synchronous save (but it's okay — it's in background)
                self.save_stats(sync=True)
            finally:
                try:
                    self._save_thread_lock.release()
                except Exception:
                    pass

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def save_stats(self, sync=False):
        """Saves stats with backup and error recovery.
           If called from main thread without sync flag, it will try to spawn background save instead.
           When sync=True it will execute the save synchronously (used by background worker).
        """
        # If caller didn't request synchronous save, spawn a background worker instead
        if not sync:
            self._spawn_background_save()
            return

        # Synchronous save (should generally only run in background thread)
        # Acquire lock with timeout to avoid blocking forever
        acquired = self.lock.acquire(timeout=2.0)
        if not acquired:
            logging.warning("save_stats: lock busy, skipping save to avoid blocking")
            return

        try:
            # Work on shallow copy to minimize time holding lock for heavy ops
            stats_copy = dict(self.stats)

            # Prune processed_deals to prevent infinite growth
            max_deals = 5000
            pd = stats_copy.get('processed_deals', [])
            if len(pd) > max_deals:
                stats_copy['processed_deals'] = pd[-max_deals:]

            stats_copy['last_save_time'] = datetime.now().isoformat()

            # Prepare JSON string first (can be CPU heavy but done while lock held — acceptable since background)
            json_text = json.dumps(stats_copy, indent=4, default=str)

            # Create backup of current file (best-effort, don't read file while holding lock for long)
            try:
                if os.path.exists(self.file_path):
                    try:
                        # we only copy file bytes (fast) — if it fails, we move on
                        with open(self.file_path, 'r') as f:
                            current_content = f.read()
                        if current_content.strip():
                            with open(self.backup_path, 'w') as f:
                                f.write(current_content)
                    except Exception as e:
                        logging.warning(f"save_stats: could not create backup: {e}")
            except Exception:
                pass

            # Atomic write using temp file
            temp_path = self.file_path + ".tmp"
            try:
                with open(temp_path, 'w') as f:
                    f.write(json_text)
                os.replace(temp_path, self.file_path)
                logging.debug(f"Stats saved successfully at {stats_copy['last_save_time']}")
            except Exception as e:
                logging.error(f"Critical error saving stats: {e}")
                # Emergency save (best-effort)
                try:
                    emergency_path = os.path.join(config.TRADE_LOG_DIR, f"emergency_save_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                    with open(emergency_path, 'w') as f:
                        f.write(json_text)
                    logging.warning(f"Emergency save completed to {emergency_path}")
                except Exception as e2:
                    logging.critical(f"Emergency save also failed: {e2}")

        finally:
            try:
                self.lock.release()
            except Exception:
                pass

    def get_summary(self):
        """Returns a summary of current performance stats."""
        # Use quick lock attempt - if busy, return light summary
        acquired = self.lock.acquire(timeout=0.5)
        try:
            if not acquired:
                return "Performance stats busy"

            try:
                total = self.stats.get('total_trades', 0)
                if total == 0:
                    return "No trades recorded yet"

                win_rate = (self.stats.get('wins', 0) / total) * 100 if total > 0 else 0.0
                net_profit = self.stats.get('gross_profit', 0.0) - self.stats.get('gross_loss', 0.0)

                return f"Trades: {total}, Win Rate: {win_rate:.1f}%, Net P&L: ${net_profit:.2f}"
            except Exception as e:
                logging.error(f"Error generating summary: {e}")
                return "Error generating summary"
        finally:
            if acquired:
                try:
                    self.lock.release()
                except Exception:
                    pass
