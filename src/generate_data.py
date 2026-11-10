import random
import pandas as pd

random.seed(42)

T = {
    "VPN": [
        ("Cannot connect to VPN", "VPN client fails with error {e} from home network.",
         "Reset VPN profile, cleared cached credentials, reinstalled client."),
        ("VPN keeps disconnecting", "VPN drops every {m} minutes while on Wi-Fi.",
         "Disabled Wi-Fi power saving and switched VPN protocol to TCP."),
    ],
    "Password": [
        ("Account locked out", "User locked out after multiple failed logins on {d}.",
         "Unlocked account in AD and enforced password reset on next login."),
        ("Password reset not working", "Self-service reset sends no email for {d} user.",
         "Corrected the mailbox attribute in AD and re-triggered the reset."),
    ],
    "Email": [
        ("Outlook not syncing", "Outlook stuck on 'Updating inbox' with error {e}.",
         "Recreated Outlook profile and rebuilt the OST file."),
        ("Mailbox full", "User cannot send mail; mailbox quota exceeded.",
         "Archived old mail and increased quota to 100 GB."),
    ],
    "Printer": [
        ("Printer offline", "Printer on floor {m} shows offline for all users.",
         "Restarted print spooler and reassigned a static IP to the printer."),
    ],
    "Software": [
        ("Cannot install software", "Install of {a} blocked, error {e}.",
         "Approved request via software catalog and pushed install through SCCM."),
        ("Application crashing", "{a} crashes on launch after Windows update.",
         "Rolled back the update and applied vendor patch."),
    ],
    "Hardware": [
        ("Laptop running slow", "Laptop very slow, high disk usage for {d} user.",
         "Disabled startup bloat, cleaned disk, upgraded RAM."),
        ("Docking station not detected", "External monitors not detected via dock.",
         "Updated dock firmware and display drivers."),
    ],
    "Access": [
        ("Need access to shared folder", "{d} user requests access to finance share.",
         "Access approved by manager and user added to the AD security group."),
    ],
}

depts = ["HR", "Finance", "Sales", "IT", "Legal", "Marketing"]
apps = ["Adobe Reader", "Zoom", "Tableau", "SAP GUI", "Teams"]

rows = []
for i in range(200):
    cat = random.choice(list(T))
    title, desc, res = random.choice(T[cat])
    f = dict(e=random.choice(["809", "619", "0x80070005", "1603"]),
             m=random.randint(2, 30), d=random.choice(depts),
             a=random.choice(apps))
    rows.append(dict(ticket_id=f"INC{1000 + i}", category=cat,
                     priority=random.choice(["P1", "P2", "P3", "P4"]),
                     title=title, description=desc.format(**f), resolution=res))

pd.DataFrame(rows).to_csv("data/tickets.csv", index=False)
print("saved", len(rows))
