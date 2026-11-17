import json
import os
import tempfile
import unittest

import numpy as np

from semantic_model_router.intent import IntentClassifier, features
from semantic_model_router.registry import ChampionChallengerRouter, ModelRegistry, register_kind, score

PROJECT = ["arregla el test que falla", "haz commit y push", "revisa el CI del repo", "refactoriza la funcion de parseo",
           "sube la version del paquete", "mira por que falla el build", "añade tests al modulo", "corrige el bug del login"]
GENERAL = ["que tiempo hara mañana", "resume esta noticia", "traduce esto al ingles", "cuando es el proximo partido",
           "pon una alarma a las ocho", "que tengo hoy en el calendario", "busca vuelos a roma", "escribe un correo a ana"]
TRIVIAL = ["gracias", "vale", "hola", "perfecto", "buenas noches", "ok genial", "jaja", "de nada"]


def data():
    texts = PROJECT + GENERAL + TRIVIAL
    labels = ["project"] * 8 + ["general"] * 8 + ["trivial"] * 8
    return texts, labels


class TestIntentClassifier(unittest.TestCase):
    def test_learns_and_generalizes_a_little(self) -> None:
        texts, labels = data()
        clf = IntentClassifier(["project", "general", "trivial"]).fit(texts, labels)
        self.assertEqual(clf.predict("arregla el test del CI")[0], "project")
        self.assertEqual(clf.predict("que tiempo hace en madrid")[0], "general")
        self.assertEqual(clf.predict("muchas gracias")[0], "trivial")
        p = clf.predict_proba(["hola"])[0]
        self.assertAlmostEqual(float(p.sum()), 1.0)

    def test_context_changes_the_decision(self) -> None:
        texts = ["sigue", "sigue", "continua", "continua"] * 3
        labels = ["project", "general", "project", "general"] * 3
        contexts = [["active_project"], ["recent_general"]] * 6
        clf = IntentClassifier(["project", "general"]).fit(texts, labels, contexts)
        self.assertEqual(clf.predict("sigue", ["active_project"])[0], "project")
        self.assertEqual(clf.predict("sigue", ["recent_general"])[0], "general")

    def test_early_stopping_records_the_best_epoch(self) -> None:
        texts, labels = data()
        clf = IntentClassifier(["project", "general", "trivial"]).fit(
            texts, labels, epochs=40, validation=(texts[::2], labels[::2], None), check_every=5
        )
        self.assertIn(clf.meta["best_epoch"], range(5, 41, 5))
        self.assertGreater(clf.meta["val_accuracy"], 0.8)

    def test_features_include_context_tokens(self) -> None:
        a, _ = features("sigue")
        b, _ = features("sigue", ["active_project"])
        self.assertTrue(set(a) < set(b))

    def test_save_load_roundtrip(self) -> None:
        texts, labels = data()
        clf = IntentClassifier(["project", "general", "trivial"]).fit(texts, labels)
        with tempfile.TemporaryDirectory() as tmp:
            clf.save(tmp)
            again = IntentClassifier.load(tmp)
            np.testing.assert_allclose(again.predict_proba(["vale"]), clf.predict_proba(["vale"]))


class TestRegistry(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.reg = ModelRegistry(self.tmp.name)
        self.texts, self.labels = data()

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def weak(self) -> IntentClassifier:
        # trained on project vs general only: never says "trivial"
        t = PROJECT + GENERAL
        return IntentClassifier(["project", "general", "trivial"]).fit(t, ["project"] * 8 + ["general"] * 8, epochs=50)

    def strong(self) -> IntentClassifier:
        return IntentClassifier(["project", "general", "trivial"]).fit(self.texts, self.labels)

    def test_first_model_becomes_champion(self) -> None:
        v = self.reg.register(self.weak(), {"data": "d1"})
        self.assertEqual(self.reg.champion_version(), v)
        self.assertEqual(self.reg.info(v)["data"], "d1")

    def test_challenger_promoted_only_if_it_wins(self) -> None:
        self.reg.register(self.weak())
        v2 = self.reg.register(self.strong())
        decision = self.reg.promote_if_better(v2, self.texts, self.labels)
        self.assertTrue(decision.promoted, decision.reasons)
        self.assertEqual(self.reg.champion_version(), v2)
        # the weak one cannot take the crown back
        v3 = self.reg.register(self.weak())
        back = self.reg.promote_if_better(v3, self.texts, self.labels)
        self.assertFalse(back.promoted)
        self.assertEqual(self.reg.champion_version(), v2)
        history = self.reg.history()
        self.assertEqual([h["promoted"] for h in history], [True, False])
        self.assertIn("macro_f1", history[0]["challenger"])

    def test_class_regression_blocks_promotion(self) -> None:
        self.reg.register(self.strong())
        v2 = self.reg.register(self.weak())
        d = self.reg.promote_if_better(v2, self.texts, self.labels, min_gain=-1.0)  # even if F1 were fine
        self.assertFalse(d.promoted)
        self.assertTrue(any("recall of 'trivial'" in r for r in d.reasons))

    def test_apply_external_scorecards(self) -> None:
        from semantic_model_router.registry import Scorecard

        self.reg.register(self.weak())
        v2 = self.reg.register(self.strong())
        champ = Scorecard("v1", 10, 0.6, 0.55, {"a": {"recall": 0.5}}, 1.0, 2.0)
        chall = Scorecard(v2, 10, 0.8, 0.75, {"a": {"recall": 0.9}}, 1.0, 50.0)
        d = self.reg.apply(v2, champ, chall, max_latency_ms=10.0)
        self.assertFalse(d.promoted)
        self.assertTrue(any("latency" in r for r in d.reasons))
        self.assertTrue(self.reg.apply(v2, champ, chall).promoted)

    def test_scorecard(self) -> None:
        card = score(self.strong(), self.texts, self.labels, version="x")
        self.assertEqual(card.n, 24)
        self.assertGreater(card.accuracy, 0.9)
        self.assertIn("trivial", card.per_class)
        self.assertGreaterEqual(card.latency_ms_p95, card.latency_ms_p50)

    def test_custom_kinds_and_shadow_router(self) -> None:
        class Constant:
            labels = ["project", "general", "trivial"]

            def predict_proba(self, texts, contexts=None):
                return np.tile([0.1, 0.8, 0.1], (len(texts), 1))

            def save(self, directory):
                os.makedirs(directory, exist_ok=True)
                with open(os.path.join(directory, "model.json"), "w") as fh:
                    json.dump({"kind": "constant"}, fh)

        register_kind("constant", lambda d: Constant())
        self.reg.register(self.strong())
        v2 = self.reg.register(Constant())
        router = ChampionChallengerRouter(self.reg, challenger=v2)
        out = router.route("arregla el test que falla")
        self.assertEqual(out["label"], "project")
        self.assertEqual(out["challenger_label"], "general")
        self.assertEqual(router.shadow_agreement, 0.0)


if __name__ == "__main__":
    unittest.main()
