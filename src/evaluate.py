import pandas as pd
from src.ingest import STRATEGIES
from src.retriever import get_retriever

EVAL = [
    ("VPN client throws error 809 from home", ["Cannot connect to VPN", "VPN troubleshooting"]),
    ("remote connection drops every few minutes over wireless", ["VPN keeps disconnecting", "VPN troubleshooting"]),
    ("user locked out after too many wrong passwords", ["Account locked out", "Account lockout policy"]),
    ("self service reset email never arrives", ["Password reset not working"]),
    ("inbox stuck updating", ["Outlook not syncing", "Outlook sync issues"]),
    ("cannot send emails, quota exceeded", ["Mailbox full", "Outlook sync issues"]),
    ("printer on the floor shows offline", ["Printer offline", "Printer offline checklist"]),
    ("installation blocked error 1603", ["Cannot install software"]),
    ("app crashes after windows update", ["Application crashing"]),
    ("pc extremely slow, disk at 100 percent", ["Laptop running slow"]),
    ("monitors not detected through dock", ["Docking station not detected"]),
    ("need access to the finance share", ["Need access to shared folder", "Shared folder access"]),
    ("error 0x80070005", ["Outlook not syncing", "Cannot install software"]),
    ("error 619", ["Cannot connect to VPN", "Outlook not syncing", "Cannot install software"]),
    ("cant tunnel into the corporate network from my house", ["Cannot connect to VPN", "VPN troubleshooting"]),
    ("coworker gets kicked out after typing wrong pass a few times", ["Account locked out", "Account lockout policy"]),
    ("pasword reset emial not comming", ["Password reset not working"]),
    ("outlook says updating forever", ["Outlook not syncing", "Outlook sync issues"]),
    ("print queue dead on 3rd floor", ["Printer offline", "Printer offline checklist"]),
    ("zoom wont open since patch tuesday", ["Application crashing"]),
    ("laptop crawling and fan is loud", ["Laptop running slow"]),
    ("second screen black when docked", ["Docking station not detected"]),
    ("request permission to the finance drive", ["Need access to shared folder", "Shared folder access"]),
    ("vpn error 0x80070005", ["Cannot connect to VPN"]),
]
MODES = ["vector", "bm25", "hybrid", "hybrid_rerank"]

rows = []
for s in STRATEGIES:
    r = get_retriever(s)
    for m in MODES:
        h1 = h4 = mrr = 0
        for q, rel in EVAL:
            docs = r.search(q, mode=m, k=4)
            ranks = [i + 1 for i, d in enumerate(docs) if d.metadata["title"] in rel]
            if ranks:
                h4 += 1
                mrr += 1 / ranks[0]
                h1 += ranks[0] == 1
        n = len(EVAL)
        rows.append(dict(chunking=s, retrieval=m, hit_at_1=round(h1 / n, 2),
                         hit_at_4=round(h4 / n, 2), mrr=round(mrr / n, 2)))

df = pd.DataFrame(rows)
print(df.to_string(index=False))
df.to_csv("eval_results.csv", index=False)
