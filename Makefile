PY := .venv/bin/python
export PYTHONPATH := src

.PHONY: setup test lint smoke train train-bg stop tensorboard random-video trained-video evaluate params plots play

setup:          ## cria .venv e instala dependências
	./scripts/setup.sh
test:           ## testes rápidos
	$(PY) -m pytest -q
lint:
	.venv/bin/ruff check src tests
smoke:          ## treino de 2 mil passos para validar o pipeline
	$(PY) -m breakout_rl.train --config configs/smoke.yaml
train:          ## treino principal em primeiro plano
	$(PY) -m breakout_rl.train --config configs/ppo_breakout.yaml --resume
train-bg:       ## treino principal em segundo plano (caffeinate + nohup)
	./scripts/train_overnight.sh configs/ppo_breakout.yaml
stop:
	./scripts/stop_training.sh ppo_breakout
tensorboard:
	./scripts/tensorboard.sh
random-video:   ## item (f)
	$(PY) -m breakout_rl.record --policy random --episodes 3 --out reports/videos/random_policy.mp4
trained-video:  ## item (k)
	$(PY) -m breakout_rl.record --policy runs/ppo_breakout/best_model/best_model.zip --episodes 2 --out reports/videos/trained_policy.mp4
evaluate:
	$(PY) -m breakout_rl.evaluate --policy runs/ppo_breakout/best_model/best_model.zip --episodes 20
params:         ## item (j)
	$(PY) -m breakout_rl.model_info
plots:
	$(PY) -m breakout_rl.plots --run runs/ppo_breakout --out reports/figures
play:           ## item (b)
	$(PY) -m breakout_rl.play
