import requests
from ipaddress import IPv4Address

start_ip = IPv4Address("192.168.1.186")
end_ip = IPv4Address("192.168.255.186")

for ip_int in range(int(start_ip), int(end_ip) + 1, 256):  # шаг в 256, чтобы менять только второй байт
    for last in range(186, 187):  # только 186-й адрес (как ты указал)
        ip = IPv4Address(ip_int + last - 186)
        url = f"http://{ip}/"
        try:
            r = requests.get(url, timeout=1)
            print(f"[+] Found site: {url} — Status: {r.status_code}")
        except requests.RequestException:
            print(f"[-] No response from: {url}")
