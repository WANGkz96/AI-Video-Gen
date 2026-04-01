# Vast Template Setup

Эта схема нужна для режима:

- создал инстанс из шаблона;
- подождал автозапуск;
- нажал кнопку в Vast Instance Portal;
- попал сразу во фронт сервиса.

## Вариант Сейчас

Самый практичный вариант сейчас:

1. Брать Vast template на базе `vastai/pytorch` или другого рекомендованного образа с `Instance Portal`.
2. Добавить порт `8090`.
3. Добавить env-переменные для `Instance Portal`.
4. Вставить `on-start` команду, которая сама клонит репозиторий и запускает deploy.

## Порт Для Фронта

Сервис сейчас поднимает backend и фронт на одном порту:

- internal port: `8090`

Поэтому для кнопки на фронт нужен именно `8090`.

## Env Для Template

Минимальный набор для шаблона:

```text
OPEN_BUTTON_PORT=8090
PORT=8090
MODELS=all
GENERATOR_BACKEND=ltx-2.3
PORTAL_CONFIG=localhost:1111:11111:/:Instance Portal|localhost:8080:18080:/:Jupyter|localhost:8080:8080:/terminals/1:Jupyter Terminal|localhost:8384:18384:/:Syncthing|localhost:6006:16006:/:Tensorboard|localhost:8090:8090:/:AI Video Gen
```

Если хочешь, чтобы верхняя кнопка `Open` в карточке инстанса вела прямо на фронт, а не в portal, оставляй:

```text
OPEN_BUTTON_PORT=8090
```

Если хочешь в первую очередь попадать в `Instance Portal`, а фронт открывать отдельной карточкой внутри него, `OPEN_BUTTON_PORT` можно не задавать, а оставить только `PORTAL_CONFIG`.

## PROVISIONING_SCRIPT

Официально Vast рекомендует для template-кастомизации именно `PROVISIONING_SCRIPT`, который указывает на удалённый shell-скрипт.

В env шаблона добавь:

```text
PROVISIONING_SCRIPT=https://raw.githubusercontent.com/WANGkz96/AI-Video-Gen/master/scripts/onstart_vast_instance.sh
```

Если нужно стартовать не `master`, а другую ветку, просто добавь ещё одну env-переменную:

```text
REPO_REF=ltx-2-3-runtime
```

## Fallback: On-Start Command

Если в конкретном template удобнее использовать именно поле `On-start script` / `On-start command`, можно оставить и такой вариант:

```bash
bash -lc "curl -fsSL https://raw.githubusercontent.com/WANGkz96/AI-Video-Gen/master/scripts/onstart_vast_instance.sh | bash"
```

Если нужно запускать не `master`, а другую ветку:

```bash
bash -lc "export REPO_REF=ltx-2-3-runtime; curl -fsSL https://raw.githubusercontent.com/WANGkz96/AI-Video-Gen/master/scripts/onstart_vast_instance.sh | bash"
```

## Что Делает Автостарт

Скрипт:

- клонирует репозиторий в `/root/work/AI-Video-Gen`;
- делает `git fetch` и checkout нужной ветки;
- запускает [`scripts/deploy_vast.sh`](../scripts/deploy_vast.sh);
- поднимает сервис на `PORT=8090`.

## Что Выставить В Template UI

- Launch mode: `Jupyter + SSH`
- Open port: `8090`
- Disk: лучше `400 GB+`
- Image: любой образ Vast с `Instance Portal`, лучше `vastai/pytorch` семейства `py312`

## HF Token

Для `ltx-2.3` нужен `HF_TOKEN` с доступом к:

- `google/gemma-3-12b-it-qat-q4_0-unquantized`

Не клади токен в публичный template. Для этого используй приватный template или account-level секреты/переменные.

## Уже Запущенный Инстанс

Если инстанс уже создан без этой конфигурации, можно вручную добавить доступ к новому приложению через `Tunnels` в `Instance Portal`, указав:

```text
http://localhost:8090
```

Но для следующего боевого инстанса лучше сделать это именно через template.
