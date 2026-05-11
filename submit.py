#!/usr/bin/env python3
# Backwards-compatible entrypoint; the maintained labless submit script lives
# at labless/submit_to_labless.py.

from labless.submit_to_labless import main


raise SystemExit(main())
