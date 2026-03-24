"""
Interactive Brokers broker interface — placeholder for Phase live trading.
Uses ib_insync for order placement and management.
"""


class IBBroker:
    def __init__(self, config: dict):
        self.host = config["ib"]["host"]
        self.port = config["ib"]["port"]
        self.client_id = config["ib"]["client_id"]
        self._ib = None

    def connect(self):
        from ib_insync import IB
        self._ib = IB()
        self._ib.connect(self.host, self.port, clientId=self.client_id)

    def disconnect(self):
        if self._ib:
            self._ib.disconnect()

    def place_order(self, symbol: str, action: str, quantity: int, order_type: str = "MKT"):
        raise NotImplementedError("Order placement coming in live trading phase.")
