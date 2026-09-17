# Scoring de propension — qui contacter en premier ?

## En bref
Une équipe commerciale ne peut pas appeler tout son portefeuille avec la même énergie. L'idée de ce projet : donner un score à chaque prospect pour savoir qui a le plus de chances de dire oui, et prioriser les appels sur ceux-là.

Je n'avais pas accès à de vraies données commerciales B2B, donc j'ai utilisé un jeu de données public (campagnes télémarketing d'une banque, ~41 000 contacts, source UCI "Bank Marketing"). Le principe est le même qu'un scoring de prospects : on propose quelque chose par téléphone, et on regarde qui accepte.

## Reproduire
```bash
pip install -r requirements.txt
python3 scripts/train_scoring_model.py
```

## Le résultat
En triant les prospects par score, un chiffre ressort : les 10 % les mieux notés convertissent 4,7 fois plus que la moyenne du portefeuille. Concrètement, appeler ce groupe en priorité change beaucoup le rendement d'une journée d'appels, par rapport à appeler au hasard.

## Une erreur que j'ai évitée
Le jeu de données contient une colonne qui indique la durée de l'appel. Problème : cette information n'existe qu'une fois l'appel terminé, elle ne peut donc pas servir à décider QUI appeler avant de décrocher. Je l'ai retirée du modèle, même si ça faisait un peu baisser les résultats affichés — sinon le modèle utilisait une information qu'on n'a pas réellement au moment de la décision.

## Ce que ces chiffres ne disent pas
Ce sont des données de banque, pas du vrai B2B/PME — la méthode se transpose, le chiffre de 4,7 non. Un vrai déploiement demanderait de vraies données commerciales et un suivi dans le temps (voir [automatisation-reporting-insee](https://github.com/ATEYABA-K/automatisation-reporting-insee) pour l'idée d'un pipeline qui se met à jour tout seul).

Alvin Kouadio
