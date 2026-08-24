# Поставка и запуск Humanoid Interface

## Поддерживаемая матрица v1

| Компонент | Windows | Linux |
|---|---|---|
| ОС | Windows 10/11 x86-64 | Ubuntu 24.04 x86-64 |
| Python | CPython 3.12 | CPython 3.12 |
| GUI | PyQt6 | PyQt6 / Qt XCB |
| Камера | OpenCV `auto`, DirectShow или MSMF | OpenCV V4L2 |
| MuJoCo | штатное отдельное GLFW-окно | штатное отдельное GLFW-окно |
| CI без дисплея | Qt `offscreen` | Qt `offscreen`, MuJoCo под Xvfb |

Версия v1 запускается из checkout репозитория и локальной `.venv`. Portable-
архив, MSI/EXE, AppImage и deb-пакет пока не входят в поставку. Исходные ресурсы
и изменяемые пользовательские данные уже разведены, чтобы добавить такие сборки
без изменения доменной логики.

## Универсальный lock

Единственный канонический lock — `uv.lock`. В `pyproject.toml` разрешение
ограничено двумя непересекающимися PEP 508 средами и одновременно требует
наличия бинарных зависимостей для обеих:

- `sys_platform == 'win32' and platform_machine == 'AMD64'`;
- `sys_platform == 'linux' and platform_machine == 'x86_64'`.

Проект требует `uv >= 0.12.5`; setup-скрипты проверяют версию до изменения
окружения и сообщают понятную команду обновления.

Обычная установка всегда выполняется командой `uv sync --locked --group dev`.
`--locked` запрещает незаметное изменение решения. Обновлять lock должен
maintainer отдельной командой `uv lock`, после чего обе CI-среды обязаны пройти
полную проверку. Документация uv: [universal resolution](https://docs.astral.sh/uv/concepts/resolution/)
и [locking/syncing](https://docs.astral.sh/uv/concepts/projects/sync/).

## Windows

1. Установить uv:

   ```powershell
   winget install --id astral-sh.uv -e
   ```

2. Подготовить окружение из любой директории:

   ```powershell
   & "C:\path\to\robot-human-interface\scripts\setup_windows.ps1"
   ```

3. Запустить основной пульт:

   ```powershell
   & "C:\path\to\robot-human-interface\scripts\run_gui.ps1"
   ```

4. Запустить проверки или legacy CLI:

   ```powershell
   & "C:\path\to\robot-human-interface\scripts\run_checks.ps1"
   & "C:\path\to\robot-human-interface\scripts\run_camera_teleop.ps1" -Source synthetic
   ```

Все PowerShell-обёртки определяют корень через `$PSScriptRoot`; текущая рабочая
директория не влияет на поиск конфигурации, моделей и встроенных видео.

## Ubuntu 24.04

Автоматическая подготовка, включая системные библиотеки:

```bash
bash /path/to/robot-human-interface/scripts/setup_ubuntu.sh
bash /path/to/robot-human-interface/scripts/run_gui.sh
```

Если системные пакеты уже поставлены контейнером или администратором:

```bash
bash scripts/setup_ubuntu.sh --skip-system-deps
```

Скрипт устанавливает следующие группы Ubuntu-зависимостей:

- Qt/XCB: `libdbus-1-3`, `libfontconfig1`, `libx11-xcb1`, `libxcb1`,
  `libxcb-cursor0`, `libxcb-glx0`, `libxcb-icccm4`, `libxcb-image0`,
  `libxcb-keysyms1`, `libxcb-randr0`, `libxcb-render0`,
  `libxcb-render-util0`, `libxcb-shape0`, `libxcb-shm0`, `libxcb-sync1`,
  `libxcb-util1`, `libxcb-xfixes0`, `libxcb-xinerama0`, `libxi6`,
  `libxkbcommon-x11-0`, `libxrender1`;
- OpenGL/MuJoCo/GLFW: `libgl1`, `libegl1`, `libgl1-mesa-dri`, `libglfw3`;
- MediaPipe/OpenCV/audio runtime: `libglib2.0-0t64`, `libportaudio2`;
- камера: `v4l-utils` (V4L2);
- имя `leonardo.local`: `avahi-daemon`, `libnss-mdns`;
- headless QA: `xvfb`, `xauth`;
- Python/installer: `python3.12-venv`, `curl`, `ca-certificates`.

Проверить устройства камеры можно командой `v4l2-ctl --list-devices`. На
Wayland PyQt6 может работать через XWayland/XCB; для полностью headless тестов
используется `QT_QPA_PLATFORM=offscreen`.

## Ресурсы и изменяемые данные

Runtime-ресурсы checkout считаются read-only и в поставке v1 разрешаются
`ResourceLocator` относительно корня репозитория независимо от текущей рабочей
директории:

- `assets/models/pose_landmarker_full.task`;
- `assets/videos/`;
- `config/`;
- `models/humanoid/`.

`MANIFEST.in` включает эти каталоги в source distribution. Wheel/portable-бандл
не объявлен готовым до появления отдельного шага копирования ресурсов и smoke-
теста установленного артефакта.

Настройки и изменяемые файлы не записываются рядом с исходным кодом. GUI должен
получать каталоги через `QStandardPaths`:

| Назначение | Qt location | Пример содержимого |
|---|---|---|
| настройки и каталог путей | `AppLocalDataLocation` | `settings.json`, `user_videos.json` |
| логи | `AppLocalDataLocation/logs` | до `5 × 5 MiB` |
| миниатюры и временный кэш | `CacheLocation` | preview thumbnails |
| экспорт оператора | выбранный пользователем путь | CSV/JSON логов |

В каталоге пользовательских видео сохраняются только абсолютные пути к
исходникам; сами видео не копируются. Недоступный путь остаётся диагностируемой
записью и не приводит к поиску относительно CWD.

`QApplication` создаётся в точке входа до ленивого импорта OpenCV и MediaPipe.
Это не даёт Qt-плагинам, поставляемым OpenCV, перехватить поиск платформенных
плагинов PyQt6.

## Лицензия PyQt6

PyQt6 распространяется по GPLv3 и коммерческой лицензии. Перед передачей
бинарной сборки третьим лицам владелец продукта должен выбрать совместимый
вариант: выполнить требования GPLv3 для всего распространяемого производного
приложения либо приобрести коммерческую лицензию PyQt. Добавление зависимости
в этот прототип само по себе не объявляет отдельную лицензию репозитория.
[Условия PyQt6 на PyPI](https://pypi.org/project/PyQt6/).

## Ручная приёмка платформы

CI проверяет установку lock-файла, pytest, synthetic headless pipeline, Qt
offscreen и MuJoCo smoke. До релиза на каждой ОС отдельно проверяются:

1. реальная камера и смена backend/разрешения;
2. открытие, управление и закрытие passive viewer MuJoCo;
3. запуск GUI из директории, не являющейся корнем проекта;
4. переключение всех трёх типов источников без зависших worker/дескрипторов;
5. реальный робот только в свободной зоне и с доступным физическим E-stop.

Программный interlock не называется E-stop и не гарантирует нейтральную позу
после разрыва WebSocket.
