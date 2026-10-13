"""Thin wrapper around Alpaca's trading API (paper trading by default)."""
from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from common.settings import Credentials


class DuplicateOrder(Exception):
    """The same order was already sent today (safe to ignore)."""


class Broker:
    def __init__(self, creds: Credentials, client=None):
        self.paper = creds.paper
        self.client = client or TradingClient(creds.api_key, creds.secret_key, paper=creds.paper)

    # ----- account -----
    def market_is_open(self) -> bool:
        return bool(self.client.get_clock().is_open)

    def equity(self) -> float:
        return float(self.client.get_account().equity)

    def position_qty(self, symbol: str) -> float:
        try:
            return float(self.client.get_open_position(symbol).qty)
        except APIError:
            return 0.0

    def position_values(self) -> dict:
        """{symbol: market value in $} for everything the account holds."""
        return {p.symbol: float(p.market_value) for p in self.client.get_all_positions()}

    # ----- orders -----
    def market_order(self, symbol, side: str, qty=None, notional=None, client_order_id=None):
        """Market order for a share count (qty) or a dollar amount (notional).

        client_order_id makes orders idempotent: if the scheduler runs twice the
        same day, Alpaca rejects the repeat instead of trading twice.
        """
        req = MarketOrderRequest(
            symbol=symbol,
            qty=qty,
            notional=None if notional is None else round(float(notional), 2),
            side=OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            client_order_id=client_order_id,
        )
        try:
            return self.client.submit_order(order_data=req)
        except APIError as e:
            if "client_order_id" in str(e).lower() and "unique" in str(e).lower():
                raise DuplicateOrder(client_order_id) from e
            raise

    def close_position(self, symbol):
        return self.client.close_position(symbol)
