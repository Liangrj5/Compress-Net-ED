# from torch_e.config import get_config
from torch_e import config
# from tf_encrypted.protocol.pond.pond import Pond
# c = config.LocalConfig()
# c.add_player('rj')
# c.add_player('stt')


# p = c.get_players('rj,stt')

# print(p[0].name)
# print(p[0].index)
# print(p[0].device_name)
# print(p[0].host)

# p1 = c.get_player('rj')

a = config.get_config()
b = a.get_player('rj')
print(b.device_name)

