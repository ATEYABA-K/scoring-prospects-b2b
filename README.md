# Scoring de propension — prioriser un effort commercial B2B

Cas d'usage IA appliqué : construire un modèle qui priorise les prospects/clients à contacter en fonction de leur probabilité réelle de conversion, plutôt que de contacter tout le portefeuille au même niveau d'effort. L'angle "Chargé de Projet Data" : pas seulement entraîner un modèle qui a un bon score technique, mais vérifier qu'il produit un **gain business mesurable et honnête**.

![Le score concentre-t-il vraiment les bons prospects ?](chart_lift_deciles.png)

**Résultat clé : les 10 % de prospects les mieux notés par le modèle convertissent à 53 %, contre 11 % en moyenne sur l'ensemble du portefeuille — un gain × 4,7.** Contacter en priorité ce décile plutôt que le portefeuille entier permettrait de concentrer l'effort commercial là où il rapporte le plus, à volume de contacts identique.

## Pourquoi ce dataset

Aucune donnée réelle de campagne B2B/Pro-PME n'est accessible publiquement. Le jeu de données [UCI "Bank Marketing"](https://archive.ics.uci.edu/dataset/222/bank+marketing) (campagnes télémarketing d'une banque portugaise, 41 188 contacts, cible = souscription à un dépôt à terme) a été choisi parce que sa structure — des contacts commerciaux avec un résultat converti/non converti — est directement transposable à un scoring de leads B2B : la méthode ne dépend pas du secteur, seulement de la forme du problème (classification binaire déséquilibrée sur des données de contact commercial).

## Méthodologie

1. **Retrait d'une fuite de données** : la variable `duration` (durée de l'appel) n'est connue qu'*après* avoir appelé le prospect. L'utiliser donnerait un score artificiellement excellent, mais inutilisable en pratique — on ne peut pas prioriser un appel avec la durée... de cet appel. Beaucoup de tutoriels sur ce dataset l'incluent par erreur ; elle est exclue ici dès le départ.
2. **Deux modèles comparés** : régression logistique (baseline interprétable, pondérée pour le déséquilibre des classes) vs Gradient Boosting (`HistGradientBoostingClassifier`).
3. **Métriques adaptées au déséquilibre** : ROC-AUC et surtout **PR-AUC** (aire sous la courbe précision-rappel) — avec seulement 11 % de conversions, une accuracy brute serait trompeuse (un modèle qui prédit toujours "non" aurait déjà 89 % d'accuracy).
4. **Analyse par décile (lift)** : le vrai test métier n'est pas le score technique mais "si je contacte les X % les mieux notés, quel est mon taux de conversion réel sur des données jamais vues à l'entraînement ?"
5. **Importance des variables au niveau métier** : calculée par permutation sur les colonnes originales (`job`, `month`...), pas sur les colonnes one-hot encodées — sinon l'importance d'une variable catégorielle se dilue artificiellement entre ses modalités et devient illisible.

## Résultats

| Modèle | ROC-AUC | PR-AUC |
|---|---|---|
| Régression logistique | 0,805 | 0,463 |
| **Gradient Boosting** | **0,814** | **0,493** |

![Courbes ROC et précision-rappel](chart_courbes_roc_pr.png)

Le Gradient Boosting est retenu (légèrement meilleur sur les deux métriques). Le lift par décile ci-dessus est calculé avec ce modèle.

## Ce que le modèle a vraiment appris (et pourquoi ça compte)

![Importance des variables](chart_importance_variables.png)

Les variables qui pèsent le plus sont macroéconomiques et temporelles (`nr.employed`, `emp.var.rate`, `cons.price.idx`, `month`, `contact`) — pas des caractéristiques individuelles du client (`age`, `default`, `poutcome` arrivent en bas de classement). **Le modèle apprend surtout "à quel moment la campagne fonctionne", pas "quel profil de client convertit."** C'est un résultat honnête à signaler plutôt qu'à cacher : un score construit sur ce jeu de données priorise en grande partie une fenêtre temporelle favorable, pas un profil de prospect. Sur un vrai cas B2B, ça voudrait dire enrichir les données avec de vraies variables comportementales/firmographiques (secteur, taille d'entreprise, historique d'achat) avant de faire confiance au score pour prioriser des *individus* plutôt qu'une *période*.

## Limites à connaître

- **Transposition, pas données réelles** : le dataset est du retail bancaire, pas du B2B/PME — la méthode est transférable, les résultats chiffrés (le facteur ×4,7) ne le sont pas tels quels.
- **Pas de coût métier modélisé** : cette analyse ne dit pas si le coût d'un contact commercial justifie de descendre jusqu'au 3ᵉ ou 4ᵉ décile — ça dépend du coût réel d'un contact et de la valeur d'une conversion, à chiffrer avec le métier.
- **Dérive dans le temps** : vu le poids des variables macroéconomiques, un modèle entraîné sur une période donnée se dégraderait probablement vite si le contexte économique change — un vrai déploiement demanderait un suivi de performance et un ré-entraînement régulier (voir le projet [automatisation-reporting-insee](https://github.com/ATEYABA-K/automatisation-reporting-insee) pour un exemple de pipeline qui automatise ce type de rafraîchissement).

## Pistes pour aller plus loin

- Modéliser le coût/bénéfice réel (coût d'un contact vs valeur d'une conversion) pour trouver le décile de coupure optimal plutôt que de s'arrêter à "le top 10 % est meilleur".
- Tester une variable "mois" traitée comme cyclique plutôt que catégorielle (décembre et janvier sont proches dans le temps, pas dans l'encodage one-hot actuel).
- Calibrer les probabilités prédites (`CalibratedClassifierCV`) si le score doit être lu comme une vraie probabilité et pas seulement comme un rang.

## Fichiers

- `data/bank-additional-full.csv` — données source (UCI)
- `data/resultats_deciles.csv` — résultats de l'analyse par décile
- `scripts/train_scoring_model.py` — pipeline complet (chargement → modèles → évaluation → graphiques)
- `chart_lift_deciles.png`, `chart_courbes_roc_pr.png`, `chart_importance_variables.png` — graphiques

## Auteur

Alvin Kouadio
