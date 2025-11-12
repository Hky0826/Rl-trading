# File: portfolio_tracker.py
# Description: (ENHANCED) Robust portfolio tracking with background IO and lock timeouts
# =============================================================================
import json
import os
import logging
import threading
from datetime import datetime
import numpy as np
import config

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif hasattr(obj, 'item'):  # other numpy scalars
        return obj.item()
    return obj

class PortfolioTracker:
    def __init__(self):
        self.file_path = os.path.join(config.TRADE_LOG_DIR, "portfolio_tracker.json")
        self.backup_path = os.path.join(config.TRADE_LOG_DIR, "portfolio_tracker_backup.json")
        self.lock = threading.Lock()  # Thread safety

        # Ensure directory exists
        os.makedirs(config.TRADE_LOG_DIR, exist_ok=True)

        self.data = self._load_data()
        self.peak_equity = self.data.get('peak_equity', config.INITIAL_EQUITY)
        self.open_trade_params = self.data.get('open_trade_params', {})

        # Track updates for auto-save
        self.updates_since_save = 0
        self.last_save_time = datetime.now()

        # Background save guard
        self._bg_save_lock = threading.Lock()

        # Ensure a file exists immediately at startup
        if not os.path.exists(self.file_path):
            try:
                self._safe_save(self.data, self.file_path)
                logging.info("Created new portfolio tracker file at startup")
            except Exception as e:
                logging.error(f"Failed to create initial portfolio tracker: {e}")

    # -------------------------
    # Loading & validation
    # -------------------------
    def _load_data(self):
        data = self._try_load_file(self.file_path)
        if data:
            return data

        logging.warning("Main portfolio tracker corrupted or missing, trying backup...")
        data = self._try_load_file(self.backup_path)
        if data:
            try:
                self._safe_save(data, self.file_path)
                logging.info("Restored portfolio tracker from backup")
            except Exception as e:
                logging.error(f"Could not restore from backup: {e}")
            return data

        logging.info("Creating new portfolio tracker")
        return self._get_initial_data()

    def _try_load_file(self, filepath):
        if not os.path.exists(filepath):
            return None

        try:
            with open(filepath, 'r') as f:
                content = f.read()
                if not content.strip():
                    return None
                data = json.loads(content)
                data = self._validate_and_repair_data(data)
                logging.info(f"Loaded portfolio tracker: Peak Equity ${data['peak_equity']:.2f}, {len(data['open_trade_params'])} open trades")
                return data
        except (json.JSONDecodeError, IOError, ValueError) as e:
            logging.error(f"Error loading {filepath}: {e}")
            return None

    def _validate_and_repair_data(self, data):
        if 'peak_equity' not in data or not isinstance(data.get('peak_equity'), (int, float)):
            data['peak_equity'] = config.INITIAL_EQUITY
            logging.warning(f"Reset peak_equity to initial value: {config.INITIAL_EQUITY}")

        if 'open_trade_params' not in data or not isinstance(data.get('open_trade_params'), dict):
            data['open_trade_params'] = {}
            logging.warning("Reset open_trade_params to empty dict")

        if data['peak_equity'] < 0:
            logging.warning(f"Invalid peak_equity {data['peak_equity']}, resetting to initial")
            data['peak_equity'] = config.INITIAL_EQUITY
        elif data['peak_equity'] > config.INITIAL_EQUITY * 1000:
            logging.warning(f"Suspiciously high peak_equity {data['peak_equity']}")

        cleaned_params = {}
        for key, params in data['open_trade_params'].items():
            try:
                str_key = str(key)
                if isinstance(params, dict):
                    if 'risk_level' in params and 'rr_profile_index' in params:
                        cleaned_params[str_key] = {
                            'risk_level': float(params['risk_level']),
                            'rr_profile_index': int(params['rr_profile_index'])
                        }
                        for k, v in params.items():
                            if k not in ['risk_level', 'rr_profile_index']:
                                cleaned_params[str_key][k] = convert_numpy_types(v)
                    else:
                        logging.warning(f"Invalid params for trade {str_key}, skipping")
                else:
                    logging.warning(f"Non-dict params for trade {str_key}, skipping")
            except Exception as e:
                logging.error(f"Error cleaning trade params {key}: {e}")

        data['open_trade_params'] = cleaned_params

        if 'last_update' not in data:
            data['last_update'] = datetime.now().isoformat()
        if 'created_at' not in data:
            data['created_at'] = datetime.now().isoformat()

        return data

    def _get_initial_data(self):
        return {
            'peak_equity': config.INITIAL_EQUITY,
            'open_trade_params': {},
            'created_at': datetime.now().isoformat(),
            'last_update': datetime.now().isoformat()
        }

    # -------------------------
    # Atomic safe save helper
    # -------------------------
    def _safe_save(self, data, filepath):
        sanitized_data = convert_numpy_types(data)
        sanitized_data['last_update'] = datetime.now().isoformat()
        temp_path = filepath + ".tmp"
        with open(temp_path, 'w') as f:
            json.dump(sanitized_data, f, indent=4, default=str)
        os.replace(temp_path, filepath)

    def _save_data(self):
        """Synchronous save; intended to be called from background thread only."""
        # Acquire lock quickly
        acquired = self.lock.acquire(timeout=1.0)
        if not acquired:
            logging.warning("_save_data: main lock busy, skipping synchronous save")
            return

        try:
            # Create shallow copy for writing
            data_copy = dict(self.data)
            # Ensure open_trade_params converted
            data_copy['open_trade_params'] = {k: convert_numpy_types(v) for k, v in self.open_trade_params.items()}
            # Backup current file (best-effort)
            try:
                if os.path.exists(self.file_path):
                    with open(self.file_path, 'r') as f:
                        current = f.read()
                    if current.strip():
                        with open(self.backup_path, 'w') as f:
                            f.write(current)
            except Exception as e:
                logging.warning(f"_save_data: could not create backup: {e}")

            # Write atomically
            try:
                self._safe_save(data_copy, self.file_path)
                self.last_save_time = datetime.now()
                self.updates_since_save = 0
                logging.debug("Portfolio tracker saved successfully")
            except Exception as e:
                logging.error(f"Critical error saving portfolio tracker: {e}")
                try:
                    emergency_path = os.path.join(config.TRADE_LOG_DIR, f"portfolio_emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
                    self._safe_save(data_copy, emergency_path)
                    logging.warning(f"Emergency save completed to {emergency_path}")
                except Exception as e2:
                    logging.critical(f"Emergency save also failed: {e2}")
        finally:
            try:
                self.lock.release()
            except Exception:
                pass

    # -------------------------
    # Background save spawn
    # -------------------------
    def _spawn_bg_save(self):
        if not self._bg_save_lock.acquire(blocking=False):
            return

        def _bg_worker():
            try:
                self._save_data()
            finally:
                try:
                    self._bg_save_lock.release()
                except Exception:
                    pass

        t = threading.Thread(target=_bg_worker, daemon=True)
        t.start()

    # -------------------------
    # Public API
    # -------------------------
    def update_peak_equity(self, current_equity):
        try:
            current_equity = float(current_equity)
        except (TypeError, ValueError):
            logging.warning(f"Invalid equity value: {current_equity}")
            return

        if current_equity <= 0 or np.isnan(current_equity):
            logging.warning(f"Invalid equity value: {current_equity}")
            return

        save_needed = False
        old_peak = None

        acquired = self.lock.acquire(timeout=0.5)
        try:
            if acquired:
                if current_equity > self.peak_equity:
                    old_peak = self.peak_equity
                    self.peak_equity = current_equity
                    self.data['peak_equity'] = self.peak_equity
                    self.updates_since_save += 1
                    save_needed = True
        finally:
            if acquired:
                try:
                    self.lock.release()
                except Exception:
                    pass

        if save_needed:
            logging.info(f"New peak equity: ${self.peak_equity:.2f} (was ${old_peak:.2f})")
            # Spawn background save (non-blocking)
            self._spawn_bg_save()

    def get_current_drawdown(self, current_equity):
        try:
            current_equity = float(current_equity)
            if current_equity <= 0:
                logging.warning(f"Invalid equity for drawdown calculation: {current_equity}")
                return 0.0
            if self.peak_equity > 0:
                drawdown = (self.peak_equity - current_equity) / self.peak_equity
                return max(0.0, min(1.0, drawdown))
            return 0.0
        except Exception as e:
            logging.error(f"Error calculating drawdown: {e}")
            return 0.0

    def add_open_trade(self, order_ticket, params):
        acquired = self.lock.acquire(timeout=0.5)
        try:
            if not acquired:
                logging.warning("add_open_trade: lock busy, updating in-memory only")
                # best-effort in-memory update without saving
                try:
                    str_ticket = str(order_ticket)
                    sanitized_params = {
                        'risk_level': float(params.get('risk_level', 0)),
                        'rr_profile_index': int(params.get('rr_profile_index', 0)),
                        'added_at': datetime.now().isoformat()
                    }
                    self.open_trade_params[str_ticket] = sanitized_params
                    self.data['open_trade_params'] = self.open_trade_params
                    return
                except Exception as e:
                    logging.error(f"Error in-memory adding trade: {e}")
                    return

            # Normal safe path
            try:
                if order_ticket is None:
                    logging.warning("Attempted to add trade with None ticket")
                    return

                str_ticket = str(order_ticket)
                if not params or not isinstance(params, dict):
                    logging.warning(f"Invalid params for trade {str_ticket}")
                    return

                if 'risk_level' not in params or 'rr_profile_index' not in params:
                    logging.warning(f"Missing required fields in params for trade {str_ticket}")
                    return

                sanitized_params = {
                    'risk_level': float(params['risk_level']),
                    'rr_profile_index': int(params['rr_profile_index']),
                    'added_at': datetime.now().isoformat()
                }

                if sanitized_params['risk_level'] not in config.RISK_LEVELS:
                    logging.warning(f"Invalid risk level {sanitized_params['risk_level']}")
                if sanitized_params['rr_profile_index'] >= len(config.RR_PROFILES):
                    logging.warning(f"Invalid RR profile index {sanitized_params['rr_profile_index']}")

                for k, v in params.items():
                    if k not in ['risk_level', 'rr_profile_index']:
                        sanitized_params[k] = convert_numpy_types(v)

                if str_ticket in self.open_trade_params:
                    logging.warning(f"Trade {str_ticket} already exists, updating params")

                self.open_trade_params[str_ticket] = sanitized_params
                self.data['open_trade_params'] = self.open_trade_params

                logging.info(f"Added trade {str_ticket} with risk={sanitized_params['risk_level']}, rr_index={sanitized_params['rr_profile_index']}")
                self.updates_since_save += 1

                if self.updates_since_save >= 5:
                    self._spawn_bg_save()
            except Exception as e:
                logging.error(f"Error adding open trade {order_ticket}: {e}")
        finally:
            if acquired:
                try:
                    self.lock.release()
                except Exception:
                    pass

    def remove_open_trade(self, order_ticket):
        acquired = self.lock.acquire(timeout=0.5)
        try:
            if not acquired:
                logging.warning("remove_open_trade: lock busy, performing best-effort in-memory removal")
                try:
                    str_ticket = str(order_ticket)
                    params = self.open_trade_params.pop(str_ticket, None)
                    self.data['open_trade_params'] = self.open_trade_params
                    return params
                except Exception as e:
                    logging.error(f"Error in-memory removing trade: {e}")
                    return None

            try:
                if order_ticket is None:
                    return None

                str_ticket = str(order_ticket)
                params = self.open_trade_params.pop(str_ticket, None)
                if params:
                    self.data['open_trade_params'] = self.open_trade_params
                    logging.info(f"Removed trade {str_ticket}")
                    self.updates_since_save += 1
                    if self.updates_since_save >= 3:
                        self._spawn_bg_save()
                    return params
                else:
                    logging.debug(f"Trade {str_ticket} not found in open trades")
                    return None
            except Exception as e:
                logging.error(f"Error removing trade {order_ticket}: {e}")
                return None
        finally:
            if acquired:
                try:
                    self.lock.release()
                except Exception:
                    pass

    def get_open_trade_count(self):
        return len(self.open_trade_params)

    def cleanup_stale_trades(self, active_tickets):
        acquired = self.lock.acquire(timeout=0.5)
        try:
            if not acquired:
                logging.warning("cleanup_stale_trades: lock busy, skipping cleanup this cycle")
                return

            try:
                active_tickets_str = {str(t) for t in active_tickets}
                stale_trades = [t for t in list(self.open_trade_params.keys()) if t not in active_tickets_str]

                if stale_trades:
                    for ticket in stale_trades:
                        del self.open_trade_params[ticket]
                    self.data['open_trade_params'] = self.open_trade_params
                    logging.info(f"Cleaned up {len(stale_trades)} stale trades")
                    self._spawn_bg_save()
            except Exception as e:
                logging.error(f"Error cleaning up stale trades: {e}")
        finally:
            if acquired:
                try:
                    self.lock.release()
                except Exception:
                    pass

    def force_save(self):
        """Force immediate save (runs in background to avoid blocking caller)."""
        self._spawn_bg_save()
