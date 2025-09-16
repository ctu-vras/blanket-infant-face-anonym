from dataclasses import dataclass


@dataclass
class SDWebUISettings:
    parameters: SDWebUIParameters

    ipv4_address: str = "127.0.0.1"
    port: str = "7861"

    @property
    def server_url(self):
        return f"http://{self.ipv4_address}:{self.port}"
