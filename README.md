# MatchScope per iPhone

L'app è pronta per essere eseguita come web app installabile. Dopo la pubblicazione apri il suo indirizzo con Safari su iPhone, tocca **Condividi** e scegli **Aggiungi a Home**.

## Avvio locale

```bash
./venv/bin/python -m streamlit run app.py
```

Per provarla dall'iPhone nella stessa rete Wi-Fi, avviala in ascolto sulla rete locale:

```bash
./venv/bin/python -m streamlit run app.py --server.address 0.0.0.0
```

Poi apri su Safari `http://IP-DEL-TUO-MAC:8501`. Il Mac deve rimanere acceso per questa modalità.

## Pubblicazione privata

Per usarla anche fuori casa, il progetto è pronto per Render: carica questa cartella in un repository Git privato e crea un nuovo servizio dal file `render.yaml`. Nel pannello di Render imposta `MATCHSCOPE_PASSWORD` come variabile segreta: l'app resterà raggiungibile dal tuo iPhone, ma protetta da password.

Render fornisce automaticamente HTTPS. Apri poi l'indirizzo dell'app in Safari e scegli **Aggiungi a Home**. Il piano gratuito sospende il servizio dopo un periodo di inattività e può richiedere circa un minuto al primo accesso.
