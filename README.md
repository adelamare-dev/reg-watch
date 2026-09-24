# RegWatch

RegWatch est un copilote de conformité réglementaire multi-agents qui interroge un corpus croisé **DORA × EU AI Act**.

> **Ne constitue pas un conseil juridique.** RegWatch est un outil d'aide à la lecture réglementaire, pas un avis juridique. Toute décision de conformité doit être validée par un professionnel qualifié.

## Réserve méthodologique

Les recouvrements entre DORA et l'EU AI Act présentés par l'outil relèvent d'une **lecture croisée** construite à partir de sources sectorielles secondaires. Ce n'est **jamais une position officielle des régulateurs** : il n'existe à ce jour aucune guidance conjointe EBA / AI Office sur ces recoupements. Les divergences et recommandations produites par le graphe doivent être lues comme des pistes d'analyse, pas comme une interprétation faisant autorité.

## État d'avancement

L'architecture cible comprend 5 couches. Voici leur état réel dans ce dépôt :

| Couche | État |
|---|---|
| 1 — RAG | Implémentée : ingestion EUR-Lex, chunking structurel, Qdrant, retrieval en parent-document |
| 2 — Serveur MCP | Implémentée : 4 outils réglementaires, Streamable HTTP, port 8300 |
| 3 — Orchestration LangGraph | Implémentée : 3 nœuds Retrieval → Analyste → Critique, boucle de reprise bornée, 3 refus explicites |
| 4 — Observabilité & évaluation | Implémentée : instrumentation Langfuse, golden dataset, pipeline d'évaluation |
| 5 — UI | Scaffold CopilotKit présent mais **non branché au graphe** — le chat ne dialogue pas encore avec l'agent Python |

355 tests passent par défaut ; 12 tests d'intégration (nécessitant un Qdrant joignable) en sont exclus.

## Prérequis et installation

- Node.js + pnpm (dépendances JS)
- Python 3.12 + [`uv`](https://docs.astral.sh/uv/) (agent Python)
- Docker (pour Qdrant)

```bash
pnpm install
```

Sur la machine de développement utilisée pour ce projet, **toute commande Python doit s'exécuter dans WSL2** : le relais `localhost` Windows → WSL2 y est cassé, ce qui rend Qdrant injoignable depuis un Python lancé côté Windows. C'est une contrainte de cet environnement précis, pas une exigence du projet — sur une machine où le réseau WSL2 fonctionne normalement, les commandes `uv` peuvent s'exécuter directement.

## Commandes

```bash
pnpm dev                    # serveur de dev Vite (frontend)
pnpm build                  # tsc -b && vite build
pnpm lint                   # eslint .
tsx server.ts               # runtime CopilotKit (terminal séparé, port 8200)

pnpm qdrant                 # docker compose up -d qdrant
pnpm ingest                 # ingestion du corpus (aussi :verify, :dry-run)
pnpm mcp                    # serveur MCP, port 8300 (aussi mcp:stdio)
pnpm graph "<question>"     # interroge le graphe avec une question

pnpm eval <dataset.jsonl>         # valide le golden dataset et exécute les assertions déterministes, sans appel LLM
pnpm eval:strict <dataset.jsonl>  # idem, en propageant la première erreur RAGAS (usage développement)

pnpm test:agent              # 355 tests
pnpm test:agent:integration   # nécessite un Qdrant joignable
```

Le stack Docker (`docker-compose.yml`) ne déclare que deux services : `qdrant` et `mcp`. L'image `mcp` n'a encore jamais été construite dans ce dépôt — seul `pnpm qdrant` est un chemin éprouvé aujourd'hui.

## Configuration

Copier `.env.example` vers `.env` à la racine du dépôt et renseigner les variables nécessaires (fournisseur d'embeddings, Qdrant, modèle de génération, etc.).

Sans les clés `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY`, le traçage Langfuse reste simplement inactif : le graphe tourne à l'identique, sans appel réseau ni erreur.

## Golden dataset

Les 15 entrées du golden dataset (`data/golden-dataset.jsonl`) sont livrées **à l'état de gabarit** (`draft: true`), questions et références encore à rédiger. Cette rédaction est délibérément **humaine, non générée par un LLM** : évaluer un système fondé sur un LLM avec un jeu de données produit par un LLM introduirait une circularité méthodologique. Le validateur du pipeline d'évaluation **refuse d'exécuter une évaluation** tant que des gabarits n'ont pas été rédigés — c'est un choix de méthode assumé, pas une limite provisoire à masquer.

## Limite méthodologique de l'évaluation

Parmi les métriques RAGAS prévues, `faithfulness` est partiellement circulaire : le nœud Critique du graphe optimise déjà l'ancrage de la réponse dans les sources, donc cette métrique mesure en partie ce que le système est conçu pour garantir. `context_precision` et `context_recall` évaluent uniquement le retrieval, une étape que le Critique ne touche pas — ce sont les mesures réellement indépendantes du pipeline.

## Licence

Double licence :

- **Code** : MIT
- **Corpus réglementaire** (textes issus d'EUR-Lex) : CC-BY-4.0, au titre de la décision 2011/833/UE

Mentions obligatoires, à reproduire fidèlement pour toute redistribution du corpus :

> © Union européenne, \<année\> — source : EUR-Lex
> Seule la version électronique publiée sur EUR-Lex fait foi juridiquement. Les textes redistribués ici ne constituent pas une version authentique.
