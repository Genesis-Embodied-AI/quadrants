# Troublshooting

## In case of crash/seg fault

- run without cache - or clear cache - to see if this resolves the issue
- if running without cache solves the seg fault, then clear the cache

To run without cache:
```
ti.init(offline_cache=False, ...)
```

To clear cache:
- the cache is located by default on linux and mac at `~/.cache/gstaichi`
- simply remove this entire folder:
```
rm -Rf ~/.cache/gstaichi
```

If this doesn't solve the problem, then you'll likely need to log a github issue, providing
as much information as possible, and crucially a minimum reproducible example, to reproduce
the seg fault.
