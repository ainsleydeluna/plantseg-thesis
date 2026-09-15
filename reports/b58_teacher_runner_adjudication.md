# B58 — TeacherRunner adjudication: G18 architecture selected, implementation still plan-gated

**Type:** scratch adjudication record and design decision. **Status:** G18 architecture **SELECTED**;
implementation **NOT AUTHORIZED** (plan-gated). G20's design-level timing conflict is **resolved by the
selected design** and stays **OPEN** pending CUDA evidence. **Date:** 2026-09-15.
**Scope:** CPU only, scratch only, every container `--network none`. No GPU, pod, checkpoint download,
training, validation run or test-split access. The only repository writes are this report and the
authorized reconciliation of B52 §11, B56 and B57.
**Repository:** `1dbd1a1`. **Image:** `plantseg-teacher:local`,
`sha256:8c1f31f6e7fc09c6de9d404cf76a647614748f6bc1765775391b4411bae9ba45`, base digest
`sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf`, mmengine 0.10.7.

> **Evidence tags used throughout.**
> **MEASURED** — raw output embedded in this record.
> **SOURCE-PROVEN** — pinned source lines printed in an embedded transcript.
> **[INFERRED]** — reasoning beyond the measurement, stated with what would confirm it.
> **PLAN / SELECTED DESIGN** — a ruled design, not implemented and not executed.
> **STILL OPEN** — not settled by this record.

## Evidence set

| Id | Content | source sha256 |
|---|---|---|
| T1 | preflight and container canary | `7addcf08bbaf1cf5317fd4e4d7a6adac783c11b9a2cccc39f822ad3c6ac5cdf2` |
| T2 | G0 — pinned MMEngine source | `3a1ff35a470a1980a6063b5049fe4993ac70438a852d6e36c2bc13f369c295b2` |
| T3 | G1 — minimal subclass, cuBLAS variable inherited | `d79a66ffde07933cf7283da7c6e3a7f2b094045b9d700cd02acac288f4e87250` |
| T4 | G2 — negative cuBLAS control | `372ce1faa6f946cef870c0400b4e8f465e2cda9b4ae82c715aa11c2d7e6b565d` |
| T5 | G3 — merged teacher config | `f8d28a3fa1a9dbbdeecb1504946f3f2f39297402d4d102c87bb194be53442d54` |
| T6 | post-run repository safety | `4942b890038cb444a6a791ba629ae82238957d218fab30c5a139d089e55284f4` |
| T7 | Chapter III wording, repo copy of `docs/reference/ch3.pdf`. Embedded with CRLF→LF normalization only; embedded sha256 `f68c0162bccb2734068ad901007efda8f669879f77f9c2d6cbf94ccbe586571d` | `a96117eb7db0767543182e749ab0c91218d462940fb1e57cbeffddf0bf85de78` |
| probe | `b58_teacher_runner.py` — scratch prototype, Appendix A | `d71279dae43bb12179ceb138b5ca4b347208bc77638f202810a40bb509b011c5` |
| probe | `b58_ch3_search.py` — text search, Appendix B | `e29cbd8672690d250f27ae72c65c9f3e6ff493ae4a44f65a1bf76926d9e7a412` |
| config | scratch reordered teacher config — the HEAD blob `9752882` (sha256 `6004209f…9cf4`) with lines 42 and 44 swapped; matches [B56 §5.3](b56_teacher_survey.md) | `510b212b0ea36782baf47980cf10884f79e6798e2332366b51f6a206d5911ab5` |

All seven transcripts come from one transcript runner, which prints `$ <command>`, then the raw stdout
and stderr, then `[exit N]`. Each is embedded whole, including the blank line that ends it, so each fenced
block closes directly after its source's final byte.

- **T1–T6 are byte-exact embeds.** The bytes inside each fence are the scratch source file's bytes, and they
  hash to the source sha256 in the table.
- **T7 is content-verbatim, with one disclosed normalization.** In its source file, lines 3–17 — the 15 lines
  printed by the Chapter III search script — end in CRLF; every other line ends in LF. The embed writes those
  15 line endings as LF. Exactly 15 CR bytes, each from a CRLF pair, are removed, and no other byte is
  transformed. The embedded bytes are `source.replace(b"\r\n", b"\n")`: source sha256
  `a96117eb7db0767543182e749ab0c91218d462940fb1e57cbeffddf0bf85de78`, embedded sha256 `f68c0162bccb2734068ad901007efda8f669879f77f9c2d6cbf94ccbe586571d`.

The byte-level proof of both statements runs against this file after it is written, so it is reported
outside this record.

**The scratch prototype is not the selected implementation.** It asserts the cuBLAS variable and applies
the policy, but it has none of the first-call CUDA preconditions or the marker that §9 selects, and its
assertion message is superseded by §10.

---

## Findings at a glance

| Item | Result | Tag |
|---|---|---|
| Subclass dispatch | `from_cfg` constructs through `cls(...)` (`runner.py:462`); `TeacherRunner.from_cfg` returns a `TeacherRunner` | SOURCE-PROVEN (T2); MEASURED (T3, T5) |
| Override scope | `TeacherRunner` defines only `set_randomness` | MEASURED (T3–T5) |
| Seeding | torch, NumPy and Python are reset to seed 42, tested against deliberate perturbation and fresh-seed controls | MEASURED (T3, T5) |
| Policy flags | deterministic algorithms on, `warn_only=True`, `cudnn.deterministic=True`, `cudnn.benchmark=False` after construction, although `env_cfg.cudnn_benchmark=True` | MEASURED (T3, T5) |
| Ordering | the policy is established at `runner.py:376`, before `_log_env` (`:403`) and `build_model` (`:429`) | SOURCE-PROVEN (T2) |
| cuBLAS variable | asserted, never created: with it absent, construction fails loudly and the variable stays unset | MEASURED (T4) |
| Resume | `resume()` re-enters `self.set_randomness` when the seeds differ (`:2060`); the surrogate call restores the full state | SOURCE-PROVEN (T2); MEASURED surrogate (T3, T5) |
| Provenance | `runner.deterministic` stays `True` and the config text keeps `deterministic=True`; no separate attestation of effective `warn_only`, effective cuDNN flags or the Runner class was found in the examined paths | MEASURED; SOURCE-PROVEN |
| G18 architecture | SELECTED; implementation plan-gated | PLAN / SELECTED DESIGN |
| G20 | design-level timing conflict resolved; CUDA evidence pending | STILL OPEN |

---

## 1. How this record was produced

The adjudication ran under the GO of 2026-09-15 ("B58 teacher-runner scratch adjudication only"): a
container canary, then G0–G3, each in a fresh CPU container with `--network none`. No repository path was
mounted. The prototype and the reordered config copy were mounted read-only from scratch. The Chapter III
search (T7) ran on the host against `docs/reference/ch3.pdf` only.

---

## 2. Preflight and container canary — MEASURED (T1)

````text
$ git rev-parse --short HEAD
1dbd1a1
[exit 0]

$ timeout 30 docker version --format 'server {{.Server.Version}} {{.Server.Os}}/{{.Server.Arch}}'
server 29.7.2 linux/amd64
[exit 0]

$ docker image inspect plantseg-teacher:local --format 'Id={{.Id}} base.digest={{index .Config.Labels "org.opencontainers.image.base.digest"}} revision={{index .Config.Labels "org.opencontainers.image.revision"}}'
Id=sha256:8c1f31f6e7fc09c6de9d404cf76a647614748f6bc1765775391b4411bae9ba45 base.digest=sha256:b80b645d6087a51bc4bae41c433ed77c3f30e43d442bf9c52c1be01698866aaf revision=1dbd1a1dcc03a70f4ef5b329be9e1b59df4ab088
[exit 0]

$ S=C:/Users/admin/AppData/Local/Temp/claude/C--Users-admin-plantseg-thesis/8a9f2677-aa42-4af2-9ebd-1d4276e3ab9a/scratchpad/snap; git status --porcelain > $S/pre_porcelain.txt && git status --porcelain --ignored=matching --untracked-files=all > $S/pre_full.txt && cat $S/pre_porcelain.txt && echo "porcelain sha256 $(sha256sum < $S/pre_porcelain.txt | cut -d' ' -f1)" && echo "full inventory $(wc -l < $S/pre_full.txt) lines sha256 $(sha256sum < $S/pre_full.txt | cut -d' ' -f1)"
 M docs/reference/reference.pdf
 M reports/b52_e1_seed42_completion.md
?? reports/b56_teacher_survey.md
?? reports/b57_strict_mode_evidence.md
porcelain sha256 f414ab9a72b5b9b465880f3e3ab7f572761dfe2db5c0731c5c9cecf14eacf331
full inventory 24 lines sha256 b4523b624573856a19f77069642f37e9b6dcce8da5756962cc8e4cc2b63210ce
[exit 0]

$ MSYS_NO_PATHCONV=1 docker run --rm --network none plantseg-teacher:local python -c "import sys, torch, mmengine; print('python', sys.version.split()[0], '| torch', torch.__version__, '| mmengine', mmengine.__version__)"
python 3.11.16 | torch 2.1.0+cu121 | mmengine 0.10.7
[exit 0]

````

HEAD, engine, image identity and base digest all match the expected baseline. The canary container started
cleanly with no mounts: Python 3.11.16, torch 2.1.0+cu121, mmengine 0.10.7. The repository baseline taken
here is compared after the run in §19.

---

## 3. G0 — pinned MMEngine source — SOURCE-PROVEN (T2)

````text
$ MSYS_NO_PATHCONV=1 docker run --rm --network none plantseg-teacher:local bash -c 'R=/usr/local/lib/python3.11/site-packages/mmengine/runner/runner.py; python -c "import mmengine; print(\"mmengine\", mmengine.__version__)"; echo "--- (1) Runner.from_cfg"; grep -n "" $R | sed -n "449,492p"; echo "--- (2) Runner.__init__ ordering"; grep -n -E "    def __init__\(|self\.setup_env\(env_cfg\)|self\._randomness_cfg = randomness|self\.set_randomness\(\*\*randomness\)|self\._log_env\(env_cfg\)|self\.model = self\.build_model\(model\)|self\.model = self\.wrap_model\(" $R; echo "--- (3) Runner.resume"; grep -n -E "    def resume\(" $R; grep -n "" $R | sed -n "2050,2060p"; echo "--- (4a) Runner.save_checkpoint, full"; grep -n "" $R | sed -n "/^[0-9]*:    def save_checkpoint(/,/^[0-9]*:    @master_only/p"; echo "--- (4b) Runner.dump_config, full"; grep -n "" $R | sed -n "/^[0-9]*:    def dump_config(/,/^[0-9]*:    def /p"; echo "--- (4c) Runner._log_env, full"; grep -n "" $R | sed -n "/^[0-9]*:    def _log_env(/,/^[0-9]*:    def /p"; echo "--- (4d) every randomness/deterministic mention in runner.py"; grep -n -E "randomness|deterministic" $R; echo "--- (4e) other provenance writers"; cd /usr/local/lib/python3.11/site-packages/mmengine && grep -n -E "pretty_text|seed|deterministic|randomness|update_info" hooks/runtime_info_hook.py; grep -n "add_config" runner/runner.py visualization/visualizer.py visualization/vis_backend.py'
mmengine 0.10.7
--- (1) Runner.from_cfg
449:
450:    @classmethod
451:    def from_cfg(cls, cfg: ConfigType) -> 'Runner':
452:        """Build a runner from config.
453:
454:        Args:
455:            cfg (ConfigType): A config used for building runner. Keys of
456:                ``cfg`` can see :meth:`__init__`.
457:
458:        Returns:
459:            Runner: A runner build from ``cfg``.
460:        """
461:        cfg = copy.deepcopy(cfg)
462:        runner = cls(
463:            model=cfg['model'],
464:            work_dir=cfg['work_dir'],
465:            train_dataloader=cfg.get('train_dataloader'),
466:            val_dataloader=cfg.get('val_dataloader'),
467:            test_dataloader=cfg.get('test_dataloader'),
468:            train_cfg=cfg.get('train_cfg'),
469:            val_cfg=cfg.get('val_cfg'),
470:            test_cfg=cfg.get('test_cfg'),
471:            auto_scale_lr=cfg.get('auto_scale_lr'),
472:            optim_wrapper=cfg.get('optim_wrapper'),
473:            param_scheduler=cfg.get('param_scheduler'),
474:            val_evaluator=cfg.get('val_evaluator'),
475:            test_evaluator=cfg.get('test_evaluator'),
476:            default_hooks=cfg.get('default_hooks'),
477:            custom_hooks=cfg.get('custom_hooks'),
478:            data_preprocessor=cfg.get('data_preprocessor'),
479:            load_from=cfg.get('load_from'),
480:            resume=cfg.get('resume', False),
481:            launcher=cfg.get('launcher', 'none'),
482:            env_cfg=cfg.get('env_cfg', dict(dist_cfg=dict(backend='nccl'))),
483:            log_processor=cfg.get('log_processor'),
484:            log_level=cfg.get('log_level', 'INFO'),
485:            visualizer=cfg.get('visualizer'),
486:            default_scope=cfg.get('default_scope', 'mmengine'),
487:            randomness=cfg.get('randomness', dict(seed=None)),
488:            experiment_name=cfg.get('experiment_name'),
489:            cfg=cfg,
490:        )
491:
492:        return runner
--- (2) Runner.__init__ ordering
62:    def __init__(self, dataset, length) -> None:
262:    def __init__(
372:        self.setup_env(env_cfg)
375:        self._randomness_cfg = randomness
376:        self.set_randomness(**randomness)
403:        self._log_env(env_cfg)
429:        self.model = self.build_model(model)
431:        self.model = self.wrap_model(
--- (3) Runner.resume
1997:    def resume(self,
2050:        # resume random seed
2051:        resumed_seed = checkpoint['meta'].get('seed', None)
2052:        current_seed = self._randomness_cfg.get('seed')
2053:        if resumed_seed is not None and resumed_seed != current_seed:
2054:            if current_seed is not None:
2055:                self.logger.warning(f'The value of random seed in the '
2056:                                    f'checkpoint "{resumed_seed}" is '
2057:                                    f'different from the value in '
2058:                                    f'`randomness` config "{current_seed}"')
2059:            self._randomness_cfg.update(seed=resumed_seed)
2060:            self.set_randomness(**self._randomness_cfg)
--- (4a) Runner.save_checkpoint, full
2147:    def save_checkpoint(
2148:        self,
2149:        out_dir: str,
2150:        filename: str,
2151:        file_client_args: Optional[dict] = None,
2152:        save_optimizer: bool = True,
2153:        save_param_scheduler: bool = True,
2154:        meta: Optional[dict] = None,
2155:        by_epoch: bool = True,
2156:        backend_args: Optional[dict] = None,
2157:    ):
2158:        """Save checkpoints.
2159:
2160:        ``CheckpointHook`` invokes this method to save checkpoints
2161:        periodically.
2162:
2163:        Args:
2164:            out_dir (str): The directory that checkpoints are saved.
2165:            filename (str): The checkpoint filename.
2166:            file_client_args (dict, optional): Arguments to instantiate a
2167:                FileClient. See :class:`mmengine.fileio.FileClient` for
2168:                details. Defaults to None. It will be deprecated in future.
2169:                Please use `backend_args` instead.
2170:            save_optimizer (bool): Whether to save the optimizer to
2171:                the checkpoint. Defaults to True.
2172:            save_param_scheduler (bool): Whether to save the param_scheduler
2173:                to the checkpoint. Defaults to True.
2174:            meta (dict, optional): The meta information to be saved in the
2175:                checkpoint. Defaults to None.
2176:            by_epoch (bool): Decide the number of epoch or iteration saved in
2177:                checkpoint. Defaults to True.
2178:            backend_args (dict, optional): Arguments to instantiate the
2179:                prefix of uri corresponding backend. Defaults to None.
2180:                New in v0.2.0.
2181:        """
2182:        if meta is None:
2183:            meta = {}
2184:        elif not isinstance(meta, dict):
2185:            raise TypeError(
2186:                f'meta should be a dict or None, but got {type(meta)}')
2187:
2188:        if by_epoch:
2189:            # self.epoch increments 1 after
2190:            # `self.call_hook('after_train_epoch)` but `save_checkpoint` is
2191:            # called by `after_train_epoch`` method of `CheckpointHook` so
2192:            # `epoch` should be `self.epoch + 1`
2193:            meta.setdefault('epoch', self.epoch + 1)
2194:            meta.setdefault('iter', self.iter)
2195:        else:
2196:            meta.setdefault('epoch', self.epoch)
2197:            meta.setdefault('iter', self.iter + 1)
2198:
2199:        if file_client_args is not None:
2200:            warnings.warn(
2201:                '"file_client_args" will be deprecated in future. '
2202:                'Please use "backend_args" instead', DeprecationWarning)
2203:            if backend_args is not None:
2204:                raise ValueError(
2205:                    '"file_client_args" and "backend_args" cannot be set at '
2206:                    'the same time.')
2207:
2208:            file_client = FileClient.infer_client(file_client_args, out_dir)
2209:            filepath = file_client.join_path(out_dir, filename)
2210:        else:
2211:            filepath = join_path(  # type: ignore
2212:                out_dir, filename, backend_args=backend_args)
2213:
2214:        meta.update(
2215:            cfg=self.cfg.pretty_text,
2216:            seed=self.seed,
2217:            experiment_name=self.experiment_name,
2218:            time=time.strftime('%Y%m%d_%H%M%S', time.localtime()),
2219:            mmengine_version=mmengine.__version__ + get_git_hash())
2220:
2221:        if hasattr(self.train_dataloader.dataset, 'metainfo'):
2222:            meta.update(dataset_meta=self.train_dataloader.dataset.metainfo)
2223:
2224:        if is_model_wrapper(self.model):
2225:            model = self.model.module
2226:        else:
2227:            model = self.model
2228:
2229:        checkpoint = {
2230:            'meta':
2231:            meta,
2232:            'state_dict':
2233:            weights_to_cpu(model.state_dict()),
2234:            'message_hub':
2235:            apply_to(self.message_hub.state_dict(),
2236:                     lambda x: hasattr(x, 'cpu'), lambda x: x.cpu()),
2237:        }
2238:        # save optimizer state dict to checkpoint
2239:        if save_optimizer:
2240:            if isinstance(self.optim_wrapper, OptimWrapper):
2241:                checkpoint['optimizer'] = apply_to(
2242:                    self.optim_wrapper.state_dict(),
2243:                    lambda x: hasattr(x, 'cpu'), lambda x: x.cpu())
2244:            else:
2245:                raise TypeError(
2246:                    'self.optim_wrapper should be an `OptimWrapper` '
2247:                    'or `OptimWrapperDict` instance, but got '
2248:                    f'{self.optim_wrapper}')
2249:
2250:        # save param scheduler state dict
2251:        if save_param_scheduler and self.param_schedulers is None:
2252:            self.logger.warning(
2253:                '`save_param_scheduler` is True but `self.param_schedulers` '
2254:                'is None, so skip saving parameter schedulers')
2255:            save_param_scheduler = False
2256:        if save_param_scheduler:
2257:            if isinstance(self.param_schedulers, dict):
2258:                checkpoint['param_schedulers'] = dict()
2259:                for name, schedulers in self.param_schedulers.items():
2260:                    checkpoint['param_schedulers'][name] = []
2261:                    for scheduler in schedulers:
2262:                        state_dict = scheduler.state_dict()
2263:                        checkpoint['param_schedulers'][name].append(state_dict)
2264:            else:
2265:                checkpoint['param_schedulers'] = []
2266:                for scheduler in self.param_schedulers:  # type: ignore
2267:                    state_dict = scheduler.state_dict()  # type: ignore
2268:                    checkpoint['param_schedulers'].append(state_dict)
2269:
2270:        self.call_hook('before_save_checkpoint', checkpoint=checkpoint)
2271:        save_checkpoint(
2272:            checkpoint,
2273:            filepath,
2274:            file_client_args=file_client_args,
2275:            backend_args=backend_args)
2276:
2277:    @master_only
--- (4b) Runner.dump_config, full
2278:    def dump_config(self) -> None:
2279:        """Dump config to `work_dir`."""
2280:        if self.cfg.filename is not None:
2281:            filename = osp.basename(self.cfg.filename)
2282:        else:
2283:            filename = f'{self.timestamp}.py'
2284:        self.cfg.dump(osp.join(self.work_dir, filename))
2285:
2286:    def _check_scheduler_cfg(
--- (4c) Runner._log_env, full
2361:    def _log_env(self, env_cfg: dict) -> None:
2362:        """Logging environment information of the current task.
2363:
2364:        Args:
2365:            env_cfg (dict): The environment config of the runner.
2366:        """
2367:        # Collect and log environment information.
2368:        env = collect_env()
2369:        runtime_env = OrderedDict()
2370:        runtime_env.update(env_cfg)
2371:        runtime_env.update(self._randomness_cfg)
2372:        runtime_env['seed'] = self._seed
2373:        runtime_env['Distributed launcher'] = self._launcher
2374:        runtime_env['Distributed training'] = self._distributed
2375:        runtime_env['GPU number'] = self._world_size
2376:
2377:        env_info = '\n    ' + '\n    '.join(f'{k}: {v}'
2378:                                            for k, v in env.items())
2379:        runtime_env_info = '\n    ' + '\n    '.join(
2380:            f'{k}: {v}' for k, v in runtime_env.items())
2381:        dash_line = '-' * 60
2382:        self.logger.info('\n' + dash_line + '\nSystem environment:' +
2383:                         env_info + '\n'
2384:                         '\nRuntime environment:' + runtime_env_info + '\n' +
2385:                         dash_line + '\n')
2386:
2387:        if self.cfg._cfg_dict:
2388:            self.logger.info(f'Config:\n{self.cfg.pretty_text}')
2389:
2390:    def _maybe_compile(self, target: str) -> None:
--- (4d) every randomness/deterministic mention in runner.py
188:        randomness (dict): Some settings to make the experiment as reproducible
189:            as possible like seed and deterministic.
193:            ``True`` in ``env_cfg`` but ``deterministic`` is ``True`` in
194:            ``randomness``, the value of ``torch.backends.cudnn.benchmark``
288:        randomness: Dict = dict(seed=None),
373:        # self._deterministic and self._seed will be set in the
374:        # `set_randomness`` method
375:        self._randomness_cfg = randomness
376:        self.set_randomness(**randomness)
487:            randomness=cfg.get('randomness', dict(seed=None)),
566:    def deterministic(self):
567:        """int: Whether cudnn to select deterministic algorithms."""
568:        return self._deterministic
698:    def set_randomness(self,
701:                       deterministic: bool = False) -> None:
708:            deterministic (bool): Whether to set the deterministic option for
709:                CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
712:                See https://pytorch.org/docs/stable/notes/randomness.html for
715:        self._deterministic = deterministic
718:            deterministic=deterministic,
2052:        current_seed = self._randomness_cfg.get('seed')
2058:                                    f'`randomness` config "{current_seed}"')
2059:            self._randomness_cfg.update(seed=resumed_seed)
2060:            self.set_randomness(**self._randomness_cfg)
2371:        runtime_env.update(self._randomness_cfg)
--- (4e) other provenance writers
51:            cfg=runner.cfg.pretty_text,
52:            seed=runner.seed,
55:        runner.message_hub.update_info_dict(metainfo)
65:        runner.message_hub.update_info('loop_stage', 'train')
66:        runner.message_hub.update_info('epoch', runner.epoch)
67:        runner.message_hub.update_info('iter', runner.iter)
68:        runner.message_hub.update_info('max_epochs', runner.max_epochs)
69:        runner.message_hub.update_info('max_iters', runner.max_iters)
71:            runner.message_hub.update_info(
83:        runner.message_hub.update_info('epoch', runner.epoch)
98:        runner.message_hub.update_info('iter', runner.iter)
130:        runner.message_hub.update_info('loop_stage', 'val')
149:                    runner.message_hub.update_info(f'val/{key}', value)
156:            runner.message_hub.update_info('loop_stage', self.last_loop_stage)
162:        runner.message_hub.update_info('loop_stage', 'test')
184:                    runner.message_hub.update_info(f'test/{key}', value)
runner/runner.py:418:            self.visualizer.add_config(self.cfg)
visualization/visualizer.py:49:      - add_configs: write config to all vis storage backends
visualization/visualizer.py:133:        >>> vis.add_config(cfg)
visualization/visualizer.py:1064:    def add_config(self, config: Config, **kwargs):
visualization/visualizer.py:1071:            vis_backend.add_config(config, **kwargs)
visualization/vis_backend.py:98:    def add_config(self, config: Config, **kwargs) -> None:
visualization/vis_backend.py:185:        >>> local_vis_backend.add_config(cfg)
visualization/vis_backend.py:233:    def add_config(self, config: Config, **kwargs) -> None:
visualization/vis_backend.py:342:        >>> wandb_vis_backend.add_config(cfg)
visualization/vis_backend.py:432:    def add_config(self, config: Config, **kwargs) -> None:
visualization/vis_backend.py:529:        >>> vis_backend.add_config(cfg)
visualization/vis_backend.py:566:    def add_config(self, config: Config, **kwargs) -> None:
visualization/vis_backend.py:654:        >>> vis_backend.add_config(cfg)
visualization/vis_backend.py:754:    def add_config(self, config: Config, **kwargs) -> None:
visualization/vis_backend.py:873:        >>> vis_backend.add_config(cfg)
visualization/vis_backend.py:918:    def add_config(self, config: Config, **kwargs) -> None:
visualization/vis_backend.py:1014:        >>> neptune_vis_backend.add_config(cfg)
visualization/vis_backend.py:1068:    def add_config(self, config: Config, **kwargs) -> None:
visualization/vis_backend.py:1156:        >>> dvclive_vis_backend.add_config(cfg)
visualization/vis_backend.py:1218:    def add_config(self, config: Config, **kwargs) -> None:
visualization/vis_backend.py:1334:        >>> aim_vis_backend.add_config(cfg)
visualization/vis_backend.py:1383:    def add_config(self, config, **kwargs) -> None:
[exit 0]

````

1. **Construction path.** `from_cfg` is a `@classmethod` (`:450`) that deep-copies the config (`:461`) and
   constructs through `cls(...)` (`:462`), passing `randomness` (`:487`) and the config (`:489`). A subclass
   therefore takes the normal path unchanged.
2. **Ordering in `Runner.__init__`.** `setup_env` (`:372`) → `_randomness_cfg` (`:375`) →
   `set_randomness` (`:376`) → `_log_env` (`:403`) → `visualizer.add_config` (`:418`) → `build_model`
   (`:429`) → `wrap_model` (`:431`). The grep's `:62` match is a different class's `__init__` in the same
   file.
3. **Resume.** `resume()` (`:1997`) calls `self.set_randomness(**self._randomness_cfg)` (`:2060`), but only
   when the checkpoint's seed differs from the configured one (`:2053`).
4. **What is recorded.**
   - Checkpoint `meta` gets `epoch` and `iter` (`:2193-2197`), `cfg=self.cfg.pretty_text` (`:2215`),
     `seed` (`:2216`), `experiment_name` (`:2217`), `time` (`:2218`) and `mmengine_version` (`:2219`),
     plus `dataset_meta` when present (`:2221-2222`). The checkpoint also carries the message hub
     (`:2234`).
   - `dump_config` writes the config to `work_dir` (`:2284`).
   - `_log_env` logs `env_cfg`, the randomness config and the seed (`:2370-2372`), then the config text
     (`:2388`).
   - `RuntimeInfoHook` puts `cfg` and `seed` into the message hub (`runtime_info_hook.py:51-52`, `:55`).
     **Lines 53–54 were not printed**, so any other keys there are not adjudicated.

---

## 4. G1 — minimal subclass, cuBLAS variable inherited — MEASURED (T3)

````text
$ MSYS_NO_PATHCONV=1 docker run --rm --network none -e CUBLAS_WORKSPACE_CONFIG=:4096:8 --mount type=bind,source=C:/Users/admin/AppData/Local/Temp/claude/C--Users-admin-plantseg-thesis/8a9f2677-aa42-4af2-9ebd-1d4276e3ab9a/scratchpad/probe,target=/probe,readonly plantseg-teacher:local python -B /probe/b58_teacher_runner.py M
arm M | torch 2.1.0+cu121 | cuda available False
callables defined on TeacherRunner itself: ['set_randomness']
[0] process start
    are_deterministic_algorithms_enabled           = False
    is_deterministic_algorithms_warn_only_enabled  = False
    cudnn.deterministic                            = False
    cudnn.benchmark                                = False
    CUBLAS_WORKSPACE_CONFIG                        = ':4096:8'
    cfg.randomness = {'seed': 42, 'deterministic': True} | cfg.env_cfg.cudnn_benchmark = True
/bin/sh: 1: gcc: not found
[1] constructed via TeacherRunner.from_cfg | type(runner).__name__ = 'TeacherRunner' | runner.seed = 42 | runner.deterministic = True
    runner._randomness_cfg = {'seed': 42, 'deterministic': True}
    flags after from_cfg returned
    are_deterministic_algorithms_enabled           = True
    is_deterministic_algorithms_warn_only_enabled  = True
    cudnn.deterministic                            = True
    cudnn.benchmark                                = False
    CUBLAS_WORKSPACE_CONFIG                        = ':4096:8'
    runner.cfg.pretty_text lines naming randomness or cudnn_benchmark: ['env_cfg = dict(cudnn_benchmark=True)', 'randomness = dict(deterministic=True, seed=42)']
[2] perturbed: deterministic algorithms off, cudnn nondeterministic, benchmark on, RNG at seed 777 + draws
    are_deterministic_algorithms_enabled           = False
    is_deterministic_algorithms_warn_only_enabled  = False
    cudnn.deterministic                            = False
    cudnn.benchmark                                = True
    CUBLAS_WORKSPACE_CONFIG                        = ':4096:8'
[3] after runner.set_randomness(**runner._randomness_cfg) | runner.seed = 42 | runner.deterministic = True
    flags
    are_deterministic_algorithms_enabled           = True
    is_deterministic_algorithms_warn_only_enabled  = True
    cudnn.deterministic                            = True
    cudnn.benchmark                                = False
    CUBLAS_WORKSPACE_CONFIG                        = ':4096:8'
[4] RNG state comparisons (torch, numpy, python):
    pre-construction perturbed == fresh seed 42          : [False, False, False]
    post-construction == fresh seed 42                    : [False, True, True]
    post-construction torch == seed 42 + one fresh build of the same model: True
    post-construction == fresh seed 43 (negative control) : [False, False, False]
    pre-resume perturbed == fresh seed 42                  : [False, False, False]
    after resume surrogate == fresh seed 42               : [True, True, True]
[exit 0]

````

- **Scope of the override:** `TeacherRunner` defines only `set_randomness`.
- **Inputs:** `CUBLAS_WORKSPACE_CONFIG=':4096:8'` at process start; `randomness={'seed': 42,
  'deterministic': True}`; `env_cfg.cudnn_benchmark=True`.
- **After construction:** `type(runner).__name__ == 'TeacherRunner'`, `runner.seed == 42`,
  `runner.deterministic == True`, and all four in-process flags match the registered policy, with the
  cuBLAS variable still `':4096:8'`.
- **Seeding, behaviourally:** the RNGs were perturbed before construction (first row, all False). Afterwards
  NumPy and Python equal a fresh seed-42 state, and torch equals a fresh seed 42 advanced by exactly one
  build of the same model. The seed-43 negative control is all False.
- **Benchmark:** `setup_env` turned `cudnn.benchmark` on from `env_cfg`, and the override turned it back off
  before construction finished.

---

## 5. G2 — negative cuBLAS control — MEASURED (T4)

````text
$ MSYS_NO_PATHCONV=1 docker run --rm --network none --mount type=bind,source=C:/Users/admin/AppData/Local/Temp/claude/C--Users-admin-plantseg-thesis/8a9f2677-aa42-4af2-9ebd-1d4276e3ab9a/scratchpad/probe,target=/probe,readonly plantseg-teacher:local python -B /probe/b58_teacher_runner.py M
arm M | torch 2.1.0+cu121 | cuda available False
callables defined on TeacherRunner itself: ['set_randomness']
[0] process start
    are_deterministic_algorithms_enabled           = False
    is_deterministic_algorithms_warn_only_enabled  = False
    cudnn.deterministic                            = False
    cudnn.benchmark                                = False
    CUBLAS_WORKSPACE_CONFIG                        = None
    cfg.randomness = {'seed': 42, 'deterministic': True} | cfg.env_cfg.cudnn_benchmark = True
[1] TeacherRunner.from_cfg raised RuntimeError: CUBLAS_WORKSPACE_CONFIG must be ':4096:8' before the interpreter starts (G8); got None
    state after the raise
    are_deterministic_algorithms_enabled           = False
    is_deterministic_algorithms_warn_only_enabled  = False
    cudnn.deterministic                            = False
    cudnn.benchmark                                = True
    CUBLAS_WORKSPACE_CONFIG                        = None
09/15 04:46:38 - mmengine - WARNING - torch.backends.cudnn.benchmark is going to be set as `False` to cause cuDNN to deterministically select an algorithm
    CUBLAS_WORKSPACE_CONFIG after also calling mmengine set_random_seed(42, deterministic=True): None
[exit 0]

````

- **Actual result:** construction failed loudly at the prototype's assertion. The variable was absent at
  launch and was not removed from inside Python.
- **Where it failed:** the process shows `cudnn.benchmark=True`, so `setup_env` (`:372`) had run and the
  raise came at `set_randomness` (`:376`). No `gcc` line appears, consistent with `_log_env` (`:403`) never
  running.
- **Not created:** the variable was still `None` after the raise, and still `None` after the process also
  called mmengine's `set_random_seed(42, deterministic=True)`.

---

## 6. G3 — merged teacher config in scratch — MEASURED (T5)

````text
$ MSYS_NO_PATHCONV=1 docker run --rm --network none -e CUBLAS_WORKSPACE_CONFIG=:4096:8 -e TEACHER_WORK_DIR=/tmp/g18_T_workdir --mount type=bind,source=C:/Users/admin/AppData/Local/Temp/claude/C--Users-admin-plantseg-thesis/8a9f2677-aa42-4af2-9ebd-1d4276e3ab9a/scratchpad/probe,target=/probe,readonly plantseg-teacher:local python -B /probe/b58_teacher_runner.py T
arm T | torch 2.1.0+cu121 | cuda available False
callables defined on TeacherRunner itself: ['set_randomness']
[0] process start
    are_deterministic_algorithms_enabled           = False
    is_deterministic_algorithms_warn_only_enabled  = False
    cudnn.deterministic                            = False
    cudnn.benchmark                                = False
    CUBLAS_WORKSPACE_CONFIG                        = ':4096:8'
    guard self-test 'NEED_TO_CONFIRM__SET_TEACHER_WORK_DIR' -> ('ABORT', 'unset sentinel')
    guard self-test '/workspace/plantseg-thesis' -> ('ABORT', '/workspace/plantseg-thesis is the repo root or beneath it')
    guard self-test '/workspace/plantseg-thesis/work_dirs/x' -> ('ABORT', '/workspace/plantseg-thesis/work_dirs/x is the repo root or beneath it')
    guard self-test 'relative/work_dir' -> ('ABORT', "relative path 'relative/work_dir'")
    guard on resolved cfg.work_dir '/tmp/g18_T_workdir' -> ('OK', '/tmp/g18_T_workdir')
    cfg.randomness = {'seed': 42, 'deterministic': True} | cfg.env_cfg.cudnn_benchmark = True
/bin/sh: 1: gcc: not found
/usr/local/lib/python3.11/site-packages/mmseg/models/builder.py:36: UserWarning: ``build_loss`` would be deprecated soon, please use ``mmseg.registry.MODELS.build()`` 
  warnings.warn('``build_loss`` would be deprecated soon, please use '
/usr/local/lib/python3.11/site-packages/mmseg/models/losses/cross_entropy_loss.py:250: UserWarning: Default ``avg_non_ignore`` is False, if you would like to ignore the certain label and average loss over non-ignore labels, which is the same with PyTorch official cross_entropy, set ``avg_non_ignore=True``.
  warnings.warn(
/usr/local/lib/python3.11/site-packages/mmseg/engine/hooks/visualization_hook.py:60: UserWarning: The draw is False, it means that the hook for visualization will not take effect. The results will NOT be visualized or stored.
  warnings.warn('The draw is False, it means that the '
[1] constructed via TeacherRunner.from_cfg | type(runner).__name__ = 'TeacherRunner' | runner.seed = 42 | runner.deterministic = True
    runner._randomness_cfg = {'seed': 42, 'deterministic': True}
    flags after from_cfg returned
    are_deterministic_algorithms_enabled           = True
    is_deterministic_algorithms_warn_only_enabled  = True
    cudnn.deterministic                            = True
    cudnn.benchmark                                = False
    CUBLAS_WORKSPACE_CONFIG                        = ':4096:8'
    runner.cfg.pretty_text lines naming randomness or cudnn_benchmark: ['cudnn_benchmark=True,', 'randomness = dict(deterministic=True, seed=42)']
[2] perturbed: deterministic algorithms off, cudnn nondeterministic, benchmark on, RNG at seed 777 + draws
    are_deterministic_algorithms_enabled           = False
    is_deterministic_algorithms_warn_only_enabled  = False
    cudnn.deterministic                            = False
    cudnn.benchmark                                = True
    CUBLAS_WORKSPACE_CONFIG                        = ':4096:8'
[3] after runner.set_randomness(**runner._randomness_cfg) | runner.seed = 42 | runner.deterministic = True
    flags
    are_deterministic_algorithms_enabled           = True
    is_deterministic_algorithms_warn_only_enabled  = True
    cudnn.deterministic                            = True
    cudnn.benchmark                                = False
    CUBLAS_WORKSPACE_CONFIG                        = ':4096:8'
[4] RNG state comparisons (torch, numpy, python):
    pre-construction perturbed == fresh seed 42          : [False, False, False]
    post-construction == fresh seed 42                    : [False, True, True]
    post-construction torch == seed 42 + one fresh build of the same model: True
    post-construction == fresh seed 43 (negative control) : [False, False, False]
    pre-resume perturbed == fresh seed 42                  : [False, False, False]
    after resume surrogate == fresh seed 42               : [True, True, True]
[exit 0]

````

- **Work-dir guard:** run after `Config.fromfile` and the probe's in-memory override. It aborted the unset
  sentinel, the repo root, a path beneath it, and a relative path, then passed the resolved external
  `cfg.work_dir` `/tmp/g18_T_workdir`.
- **Construction:** `TeacherRunner.from_cfg` built the merged teacher Runner with no data access and no
  checkpoint load; no `train()`, `val()`, `test()` or dataset iteration ran.
- **Results:** identical to G1 — `TeacherRunner`, seed 42, `deterministic=True`, all four policy flags, the
  inherited cuBLAS variable. The torch control used one fresh build of the merged SegNeXt-B model config.
- **Provenance:** the Runner's config text still reads `randomness = dict(deterministic=True, seed=42)`,
  and it also still carries the requested `cudnn_benchmark=True`.
- **Warnings during construction:** `gcc: not found` (from `collect_env`), mmseg's `build_loss`
  deprecation, the CE `avg_non_ignore` default, and VisualizationHook's `draw is False`. None is a
  determinism alert.

---

## 7. Resume-path surrogate — MEASURED (T3, T5); dispatch SOURCE-PROVEN (T2)

In both G1 and G3, the flags were deliberately perturbed (deterministic algorithms off, cuDNN
nondeterministic, benchmark on) and the RNGs moved to seed 777 plus draws. Calling
`runner.set_randomness(**runner._randomness_cfg)` then restored seed 42 for torch, NumPy and Python,
deterministic algorithms on, `warn_only=True`, `cudnn.deterministic=True`, `cudnn.benchmark=False`, and
left the cuBLAS variable at its inherited value (rows [2], [3] and the last two rows of [4] in T3 and T5).

This is a behavioural surrogate only. That a real differing-seed resume makes this call is the
SOURCE-PROVEN part (`runner.py:2053-2060`, T2). No real resume was executed.

---

## 8. Provenance — the examined paths

**Preserved:** the examined checkpoint, config and runtime paths keep `cfg.pretty_text` — which contains
`randomness.deterministic=True` — and the seed. The Runner itself reports `runner.deterministic == True`
(T3, T5).

**Not found in the examined provenance paths:** a separate attestation of the effective `warn_only` state,
the effective cuDNN flags, or the custom Runner class. The examined paths are the checkpoint `meta` and
message hub, `dump_config`, `_log_env` and `RuntimeInfoHook` (T2).

**Coverage limit:** `runtime_info_hook.py` lines 53–54 were not printed, so a framework-wide absence claim
is not established.

**One requested value differs from the effective value.** The config text and `_log_env` (`:2370`) carry
`cudnn_benchmark=True`, while the effective flag after randomness handling is `False` (T5). MMEngine's own
strict path produces the same split, because its deterministic branch also resets benchmark to `False`
([B57 E-3](b57_strict_mode_evidence.md)).

**Consequence for G18:** effective flags and the Runner class must be attested by the run record (§11),
not reconstructed from checkpoint metadata.

---

## 9. G18 — ARCHITECTURE SELECTED; IMPLEMENTATION STILL PLAN-GATED

### 9.1 Selected design — PLAN / SELECTED DESIGN

- A minimal `TeacherRunner` subclass, constructed through the normal `TeacherRunner.from_cfg(cfg)` path.
- It overrides only `set_randomness()`.
- The authoritative config stays `randomness.seed = 42` and `randomness.deterministic = True`.
- Seeding goes through MMEngine's own utility, without its strict deterministic branch.
- The override explicitly establishes `cudnn.deterministic = True`, `cudnn.benchmark = False` and
  `torch.use_deterministic_algorithms(True, warn_only=True)`.
- `CUBLAS_WORKSPACE_CONFIG` is asserted, never created by G18. G8 is its primary provider.

### 9.2 First-call rule — PLAN / SELECTED DESIGN

On the first `set_randomness()` invocation:

1. assert `torch.cuda.is_initialized()` is `False`;
2. assert the Python-visible `CUBLAS_WORKSPACE_CONFIG == ':4096:8'`;
3. reseed torch, NumPy and Python;
4. establish the registered deterministic flags;
5. assert `torch.cuda.is_initialized()` is **still** `False`;
6. only then set the private marker recording that the registered policy was established before CUDA
   initialization.

The marker is set **last**. A call that fails at any earlier step must not leave it set.

### 9.3 Later re-entry — PLAN / SELECTED DESIGN

On later `set_randomness()` calls:

- require the pre-CUDA marker;
- do **not** require CUDA to remain uninitialized;
- re-check the Python-visible cuBLAS value;
- reseed;
- re-apply the same deterministic flags.

This supports MMEngine's differing-seed resume dispatch (`runner.py:2053-2060`). Official teacher runs are
not planned to resume.

### 9.4 What the scratch adjudication did and did not exercise

- **Exercised (MEASURED):** dispatch, override scope, seeding, flag establishment, the absent-variable
  failure, and the surrogate re-entry.
- **Not exercised:** the §9.2 CUDA preconditions and marker, which the prototype does not contain. On a CPU
  host `torch.cuda.is_initialized()` is always `False`, so a CPU run could not show them anyway — including
  the case §9.3 exists for, re-entry after CUDA initialization.

### 9.5 Alternatives rejected by ruling — PLAN

- **Pre-seeding the policy and passing `randomness.deterministic=False` plus
  `env_cfg.cudnn_benchmark=False`.** It preserves the runtime flags and seeding, but MMEngine's Runner state
  and every saved config and checkpoint would then report determinism as disabled. Not selected. Its
  supporting CPU probe was run and reported on 2026-09-14; that transcript is not embedded in this record,
  so the rejection is recorded as a ruling, not as MEASURED.
- **A registered custom runner via `runner_type`.** It adds registry plumbing and configuration surface
  without solving anything `TeacherRunner.from_cfg` does not already solve. Not selected.
- **A one-shot `warn_only=True` override after `Runner` construction and before `train()`**
  ([B57 H-E6](b57_strict_mode_evidence.md)). It lands after the Runner's environment logging can initialize
  CUDA (`_log_env`, `runner.py:403`, which calls `collect_env()` at `:2368`, T2). Contract B6 requires the
  measures before CUDA initialization. It is also not resume-safe. Not selected.

---

## 10. cuBLAS claim boundary

**The G18 assertion establishes only this:** the Python-visible `os.environ['CUBLAS_WORKSPACE_CONFIG']`
value at `set_randomness` — and, on the first call, that value before the measured CUDA-init boundary.

**It does not directly inspect:**

- cuBLAS's internal workspace state;
- the C-level environment independently of Python.

**Process-start provenance comes from G8:** the image `ENV` declaration plus the launcher-entry echo.
Python's view and the C-level environment are expected to agree for image-supplied and process-start values
and for Python-side assignments. Direct mutation of the environment by native code is outside this
assertion's coverage.

**Assertion message for the future implementation:**

```text
CUBLAS_WORKSPACE_CONFIG must already equal ':4096:8' before CUDA initialization.
```

The scratch prototype's wording, "before the interpreter starts" (Appendix A), is **not** retained as the
implementation text.

---

## 11. G18 provenance plan — PLAN

**Step 0 — external process.** Record the image ID and digest, the Git commit, the checkout source-file
hash, and the container's process-start `CUBLAS_WORKSPACE_CONFIG`. Step 0 runs in other processes, and its
own GPU check initializes CUDA there, so it cannot attest the launcher's CUDA-initialization state.

**G18 module import.** When the G18 module loads, it resolves its imported `__file__`, hashes that file
immediately, and records the resolved path and sha256. This is the imported-code attestation. A checkout
file hash is not a substitute: [B55 §4](b55_teacher_acquisition.md) records a mounted volume shadowing the
image's source tree.

**Launcher process.**

- At the earliest launcher-controlled point, preferably before importing torch or MMEngine: record the
  inherited `CUBLAS_WORKSPACE_CONFIG`.
- At the first `set_randomness`: record `torch.cuda.is_initialized()` before establishing the policy,
  perform the §9.2 sequence, and record the successful post-establishment CUDA-not-initialized check.

**First `before_train_iter` and first `before_val_iter`.** Record deterministic algorithms enabled, the
`warn_only` state, `cudnn.deterministic`, `cudnn.benchmark`, `CUBLAS_WORKSPACE_CONFIG`, and the fully
qualified Runner class.

---

## 12. CUDA canary design — P-1N and P-1G8 — PLAN (not executed)

P-1N and P-1G8 are one mandatory paired mechanism test on the same `bmm`:

- **P-1N:** the cuBLAS variable is absent at process launch, and strict mode must RAISE. This proves the
  operation actually exercises the check.
- **P-1G8:** the variable is delivered by the G8 image, with no `docker run -e`. NO ALERT is interpretable
  only because P-1N first proved the detector.

**P-1N and P-1G8 are MECHANISM EVIDENCE, not the official run's own provenance.** The official teacher run
separately requires its own launcher-entry cuBLAS echo, its own first-call G18 assertion, and its own
first-train and first-validation flag records.

---

## 13. Pre-checkpoint first-forward canary — PLAN (not executed)

`before_train_iter` fires only inside the Runner loop, so this canary takes on P-5A's safeguards. If it runs
before the real checkpoint exists:

- use an explicit, canary-only `load_from=None` override, and record it;
- the dataset must be available;
- after all config overrides, resolve `cfg.work_dir` with symlinks followed. It must be explicit and
  external to the repository — not the unset sentinel, not the repo root, and not beneath it;
- set `default_hooks.checkpoint = None`. **Never delete the key** as the disabling mechanism: deleting it
  restores MMEngine's default `CheckpointHook(interval=1)`;
- immediately after `Runner.from_cfg` and **before** `train()` or `val()`, inspect the registered hooks and
  abort if any checkpoint hook remains.

After the canary:

- search the external `cfg.work_dir` for `*.pth` and for `last_checkpoint`;
- compare pre- and post-run repository inventories including ignored files, using
  `git status --porcelain --ignored=matching --untracked-files=all`, by exact inventory hash;
- do not use a broad `git diff`.

**This canary can establish development-time CUDA behaviour only.** The official checkpoint run must repeat
the first-train-iteration and first-validation-iteration attestations.

---

## 14. P-4 non-gating; P-2 and P-3 as G15 investigation — PLAN

**P-4 is NON-GATING** for official fresh runs. It stays relevant only as a diagnostic for the exceptional
differing-seed resume path, where the deterministic flags may be re-applied after CUDA initialization. It
is not a prerequisite for G18 implementation or for the official fresh teacher run.

**P-2 and P-3 serve G15's operational-consequence investigation**, not the G18 decision:

- **P-2** runs only after P-1N has proved the detector. Its three legitimate observations are WARN
  (`warn_only=True` downgrades the cuBLAS alert), RAISE (not downgraded) and NO ALERT (the arm failed to
  exercise the condition).
- **P-3** has three arms:
  - P-3C: a direct `F.cross_entropy(…, reduction='mean', ignore_index=255)` control;
  - P-3I: the same inputs with `reduction='none'`, then an explicit mean;
  - P-3M: the pinned mmseg `CrossEntropyLoss` call site, built from the operative config text.

  All three run a full backward.

---

## 15. G20 — design-level timing conflict resolved; CUDA evidence still pending

**G20 — DESIGN-LEVEL TIMING CONFLICT RESOLVED BY THE SELECTED G18 ARCHITECTURE; CUDA EVIDENCE STILL
PENDING.**

- **The requirement:** contract B6 (`docs/IMPLEMENTATION_CONTRACT.md:302-303`) requires all four measures
  "before CUDA init", with no qualifier. Chapter III's reproducibility row says "before CUDA initialization
  where applicable" (§16).
- **How the design meets it:** the policy is now established before CUDA initialization (§9.2), which
  removes the conflict the rejected post-construction override created (§9.5).
- **Why it stays OPEN:** G20 closes only when CUDA evidence shows the expected effective state at the first
  teacher forward, under the actual G18 implementation. A development canary does not replace the official
  run's own first-train and first-validation attestation.

---

## 16. Chapter III wording — MEASURED (T7)

*Representation:* content-verbatim, not byte-exact. Lines 3–17 of this block ended in CRLF in the source and
are embedded with LF, as disclosed under Evidence set. No other byte differs.

````text
$ git ls-files -s -- docs/reference/ch3.pdf && git status --porcelain -- docs/reference/ch3.pdf && PYTHONIOENCODING=utf-8 C:/Users/admin/anaconda3/python.exe -W ignore C:/Users/admin/AppData/Local/Temp/claude/C--Users-admin-plantseg-thesis/8a9f2677-aa42-4af2-9ebd-1d4276e3ab9a/scratchpad/probe/b58_ch3_search.py
100644 c38075a7c8b0693fdb350a09a5426e14589c084e 0	docs/reference/ch3.pdf
file docs/reference/ch3.pdf | bytes 484152 | sha256 8a558e8222e536d298cd330ec757fc37f988d5d37d211f8d8dc4ec952bfe1950
pdf pages 77
[CUBLAS_WORKSPACE_CONFIG] pdf page 38 | spaced hits 0 | whitespace-free hits 1 | last line '121'
[CUBLAS_WORKSPACE_CONFIG] totals: spaced 0 | whitespace-free 1
[will also remain the same] pdf page 14 | spaced hits 1 | whitespace-free hits 1 | last line '97'
    ...normalizationparameters,andmaskformattingwillbeusedforeveryexperimentalcondition throughoutallstages.Theenvironment,thesetofmetricsused,andthetypesandlevelsof corruption will also remain the same. Alltrainingrunsuserandomseed42bydefault;if multi-seed validation is ...
[will also remain the same] totals: spaced 1 | whitespace-free 1
[seed 42] pdf page 14 | spaced hits 0 | whitespace-free hits 1 | last line '97'
[seed 42] pdf page 38 | spaced hits 1 | whitespace-free hits 1 | last line '121'
    ...sed as the starting checkpoint for E1, which then becomes the starting point for E2 and E3. PyTorch/NumPy/Python RNG seeding Reproducibility All training runs use random seed 42 across torch, numpy, and python's random module. torch.backends.cudnn...
[seed 42] pdf page 44 | spaced hits 0 | whitespace-free hits 1 | last line '127'
[seed 42] totals: spaced 1 | whitespace-free 3
[before CUDA initialization where applicable] pdf page 38 | spaced hits 1 | whitespace-free hits 1 | last line '121'
    ...n.determ inistic = True, torch.backends.cudnn.bench mark = False, torch.use_deterministic_algo rithms(True, warn_only=True), and CUBLAS_WORKSPACE_C ONFIG=:4096:8 are set before CUDA initialization where applicable. Reproducibility is reported with the caveat that small 121 ...
[before CUDA initialization where applicable] totals: spaced 1 | whitespace-free 1
[exit 0]

````

- **Repo copy:** `docs/reference/ch3.pdf` — git blob `c38075a7…`, sha256 `8a558e82…`, unmodified in the
  working tree (its `git status --porcelain` line is empty).
- **p. 97 (PDF page 14):** the passage states that "the environment, the set of metrics used, and the types
  and levels of corruption will also remain the same", and that all training runs use random seed 42 "by
  default". This page's text layer drops inter-word spaces, so the seed-42 clause matched only
  whitespace-free.
- **p. 121 (PDF page 38):** the reproducibility row states seed 42 across torch, NumPy and Python's `random`,
  and that the cuDNN, deterministic-algorithm and cuBLAS settings "are set before CUDA initialization where
  applicable". The positive control `CUBLAS_WORKSPACE_CONFIG` hits the same page.
- **Authority:** contract B6 omits "where applicable". Under the authority order the implementation
  contract governs, so G20 is assessed against B6.

---

## 17. Limits and inferences

- **CPU only.** Not shown here: the flags governing actual CUDA dispatch, cuBLAS reading the variable, and
  `collect_env`'s CUDA-device branch, which does not run on CPU. The CUDA-init ordering rests on T2's
  source order.
- **Prototype, not implementation.** The §9.2 CUDA checks and marker were not exercised (§9.4), and the
  `deterministic=False` branch of the prototype was not exercised either.
- **Scratch config.** G3 used the reordered scratch copy, because HEAD's config cannot load until G16.
- **Guard root.** The G3 guard's repo root is the container checkout, `/workspace/plantseg-thesis`. The host
  repository was never mounted.
- **Reset timing.** No instrumentation sat inside `set_randomness`. The reset is shown by the
  post-construction states against controls, together with T2's source order.
- **Resume.** Only the surrogate ran; a real resume did not.
- **Coverage.** `runtime_info_hook.py:53–54` were not printed (§8).
- **cuBLAS assertion coverage** is bounded as in §10.
- **Rejected-alternative evidence.** The `deterministic=False` probe transcript is not embedded (§9.5).

---

## 18. Remaining open gates, locked sequence, and checkpoint readiness

**Locked sequence:**

```text
B58 → G16+G17 → G8 → G18 implementation → P-1N + P-1G8 (image) + CUDA first-forward attestation
    → checkpoint readiness → G2 / P-5B → teacher fine-tune
```

Survey items C and D (the SyncBN walk and the NMF double build) remain separate CPU work, immediately after
G16.

**STILL OPEN:**

- **G16 + G17:** the config reorder and the smoke loader check. Governed, plan-gated.
- **G8:** the cuBLAS `ENV` in `Dockerfile.teacher`. PRIMARY, to ship.
- **G18:** implementation of the selected architecture. Plan-gated.
- **G19:** the mmengine row in contract B6. Governed, plan-gated.
- **G20:** CUDA first-forward attestation pending.
- **G15:** the complete strict-mode op set is not determinable from artifacts.
- **G2:** teacher VRAM, via P-5B.
- **G12:** the image push to a registry.
- **The checkpoint:** four `NEED_TO_CONFIRM` fields in `docs/teacher_init_source.md`.

**Checkpoint readiness** is the [B55 §7.1](b55_teacher_acquisition.md) step-7 sequence:

1. the gated `mim` download;
2. hash verification;
3. the initialization and load test;
4. the 116-class key audit ([B55 Appendix B](b55_teacher_acquisition.md)).

---

## 19. Repository safety — MEASURED (T6)

````text
$ S=C:/Users/admin/AppData/Local/Temp/claude/C--Users-admin-plantseg-thesis/8a9f2677-aa42-4af2-9ebd-1d4276e3ab9a/scratchpad/snap; git rev-parse --short HEAD && git status --porcelain > $S/post_porcelain.txt && git status --porcelain --ignored=matching --untracked-files=all > $S/post_full.txt && echo "porcelain sha256 $(sha256sum < $S/post_porcelain.txt | cut -d' ' -f1)" && echo "full inventory $(wc -l < $S/post_full.txt) lines sha256 $(sha256sum < $S/post_full.txt | cut -d' ' -f1)" && cmp $S/pre_porcelain.txt $S/post_porcelain.txt && echo "porcelain cmp: identical" && cmp $S/pre_full.txt $S/post_full.txt && echo "full inventory cmp: identical"
1dbd1a1
porcelain sha256 f414ab9a72b5b9b465880f3e3ab7f572761dfe2db5c0731c5c9cecf14eacf331
full inventory 24 lines sha256 b4523b624573856a19f77069642f37e9b6dcce8da5756962cc8e4cc2b63210ce
porcelain cmp: identical
full inventory cmp: identical
[exit 0]

````

- **Adjudication run:** the repository's status inventory was identical before and after, by
  `git status --porcelain` and the full inventory including ignored files (`f414ab9a…` and `b4523b62…`
  both times). That instrument covers which paths are modified, untracked or ignored, not the content of
  paths that were already dirty. No container mounted a repository path (§1).
- **Containers:** every one ran with `--rm` and `--network none`.
- **Reconciliation pass:** the pass that wrote this report changes only the four authorized report paths.
  Its post-write verification is reported outside this record, because a report cannot embed its own
  post-write check.

---

## Appendix A — `b58_teacher_runner.py` (scratch prototype; not the selected implementation)

Ran unmodified from a read-only mount at `/probe` in T3–T5. Its assertion message ("before the interpreter
starts") is superseded by §10, and it contains none of §9.2's CUDA preconditions or marker.

sha256 `d71279dae43bb12179ceb138b5ca4b347208bc77638f202810a40bb509b011c5` · 193 lines · 8361 bytes

````python
#!/usr/bin/env python3
"""B58 G1-G3 TeacherRunner adjudication -- scratch only, CPU only. Raw values; no PASS/FAIL labels.

TeacherRunner overrides ONLY Runner.set_randomness(). It (1) asserts the CUBLAS_WORKSPACE_CONFIG the
process inherited, and never sets it; (2) keeps _deterministic exactly as passed; (3) seeds through
MMEngine's own set_random_seed with its strict branch disabled; (4) when deterministic is True, applies
the registered policy: cudnn.deterministic=True, cudnn.benchmark=False,
use_deterministic_algorithms(True, warn_only=True).

No measurement subclass: the proof path constructs TeacherRunner itself through from_cfg.

Behavioural seeding test: RNG states are deliberately perturbed BEFORE construction. Reference states
(fresh seed 42, fresh seed 43, and seed 42 advanced by exactly one fresh build of the same model) are
computed only at the END, because computing them reseeds the global generators.

Arms, each in a fresh interpreter:
  M  minimal Config: randomness=dict(seed=42, deterministic=True), env_cfg=dict(cudnn_benchmark=True)
  T  merged teacher config from the scratch reordered copy of the HEAD config (HEAD cannot load: G16)
"""
import sys

sys.dont_write_bytecode = True

import copy
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from mmengine.config import Config
from mmengine.registry import MODELS
from mmengine.runner import Runner, set_random_seed

SENTINEL = "NEED_TO_CONFIRM__SET_TEACHER_WORK_DIR"
REPO_ROOT = Path("/workspace/plantseg-thesis")  # the image's checkout: the operational repo root on a pod


class TeacherRunner(Runner):
    """Prototype of the G18 seam: override only set_randomness()."""

    def set_randomness(self, seed, diff_rank_seed: bool = False, deterministic: bool = False) -> None:
        inherited = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
        if inherited != ":4096:8":
            raise RuntimeError("CUBLAS_WORKSPACE_CONFIG must be ':4096:8' before the interpreter starts "
                               f"(G8); got {inherited!r}")
        self._deterministic = deterministic
        self._seed = set_random_seed(seed=seed, deterministic=False, diff_rank_seed=diff_rank_seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)


@MODELS.register_module()
class ProbeNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 2)


def flags():
    return {
        "are_deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "is_deterministic_algorithms_warn_only_enabled": torch.is_deterministic_algorithms_warn_only_enabled(),
        "cudnn.deterministic": torch.backends.cudnn.deterministic,
        "cudnn.benchmark": torch.backends.cudnn.benchmark,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def show(tag, f):
    print(tag)
    for k, v in f.items():
        print(f"    {k:46s} = {v!r}")


def rng_states():
    return (torch.get_rng_state().clone(), np.random.get_state(), random.getstate())


def states_equal(a, b):
    return [torch.equal(a[0], b[0]),
            a[1][0] == b[1][0] and np.array_equal(a[1][1], b[1][1]) and tuple(a[1][2:]) == tuple(b[1][2:]),
            a[2] == b[2]]


def seeded_states(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return rng_states()


def check_work_dir(raw):
    """Pure check, no filesystem writes: ABORT for the sentinel, a relative path, the repo root or below."""
    if raw == SENTINEL:
        return ("ABORT", "unset sentinel")
    p = Path(raw)
    if not p.is_absolute():
        return ("ABORT", f"relative path {raw!r}")
    resolved, root = p.resolve(), REPO_ROOT.resolve()
    if resolved == root or root in resolved.parents:
        return ("ABORT", f"{resolved} is the repo root or beneath it")
    return ("OK", str(resolved))


arm = sys.argv[1]
print("arm", arm, "| torch", torch.__version__, "| cuda available", torch.cuda.is_available())
print("callables defined on TeacherRunner itself:",
      sorted(k for k, v in vars(TeacherRunner).items() if callable(v)))
show("[0] process start", flags())

if arm == "M":
    cfg = Config(dict(model=dict(type="ProbeNet"), work_dir=tempfile.mkdtemp(prefix="g18_M_"),
                      randomness=dict(seed=42, deterministic=True), env_cfg=dict(cudnn_benchmark=True),
                      log_level="WARNING"))
elif arm == "T":
    cfg = Config.fromfile("/probe/cfg/segnext_mscan-b_1xb16-adamw-40k_plantseg116-512x512.py")
    cfg.log_level = "WARNING"  # probe-only override: suppress the INFO environment and config dump
    for raw in (SENTINEL, str(REPO_ROOT), str(REPO_ROOT / "work_dirs" / "x"), "relative/work_dir"):
        print("    guard self-test", repr(raw), "->", check_work_dir(raw))
    verdict = check_work_dir(cfg.work_dir)
    print("    guard on resolved cfg.work_dir", repr(cfg.work_dir), "->", verdict)
    if verdict[0] != "OK":
        raise SystemExit("STOP: work_dir guard aborted before Runner.from_cfg")
else:
    raise SystemExit(f"unknown arm {arm!r}")
print("    cfg.randomness =", dict(cfg.randomness), "| cfg.env_cfg.cudnn_benchmark =",
      cfg.env_cfg.get("cudnn_benchmark"))

# Deliberate perturbation BEFORE construction.
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)
torch.rand(8), np.random.rand(8), random.random()
pre_construction = rng_states()

try:
    runner = TeacherRunner.from_cfg(cfg)
except RuntimeError as e:
    print("[1] TeacherRunner.from_cfg raised RuntimeError:", e)
    show("    state after the raise", flags())
    set_random_seed(42, deterministic=True)
    print("    CUBLAS_WORKSPACE_CONFIG after also calling mmengine set_random_seed(42, deterministic=True):",
          repr(os.environ.get("CUBLAS_WORKSPACE_CONFIG")))
    raise SystemExit(0)
post_construction = rng_states()

print(f"[1] constructed via TeacherRunner.from_cfg | type(runner).__name__ = {type(runner).__name__!r} "
      f"| runner.seed = {runner.seed!r} | runner.deterministic = {runner.deterministic!r}")
print("    runner._randomness_cfg =", dict(runner._randomness_cfg))
show("    flags after from_cfg returned", flags())
print("    runner.cfg.pretty_text lines naming randomness or cudnn_benchmark:",
      [ln.strip() for ln in runner.cfg.pretty_text.splitlines() if "randomness" in ln or "cudnn_benchmark" in ln])

# Resume-path surrogate: deliberate flag and RNG perturbation, then re-enter set_randomness with the
# stored inputs, as Runner.resume does (runner.py:2059-2060).
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.use_deterministic_algorithms(False)
torch.manual_seed(777)
np.random.seed(777)
random.seed(777)
torch.rand(8), np.random.rand(8), random.random()
pre_resume = rng_states()
show("[2] perturbed: deterministic algorithms off, cudnn nondeterministic, benchmark on, RNG at seed 777 + draws",
     flags())
runner.set_randomness(**runner._randomness_cfg)
after_resume = rng_states()
print(f"[3] after runner.set_randomness(**runner._randomness_cfg) | runner.seed = {runner.seed!r} "
      f"| runner.deterministic = {runner.deterministic!r}")
show("    flags", flags())

# References, computed last.
ref42 = seeded_states(42)
ref43 = seeded_states(43)
torch.manual_seed(42)
if arm == "M":
    ProbeNet()
else:
    MODELS.build(copy.deepcopy(runner.cfg.model))
ref42_plus_model_build_torch = torch.get_rng_state().clone()

print("[4] RNG state comparisons (torch, numpy, python):")
print("    pre-construction perturbed == fresh seed 42          :", states_equal(pre_construction, ref42))
print("    post-construction == fresh seed 42                    :", states_equal(post_construction, ref42))
print("    post-construction torch == seed 42 + one fresh build of the same model:",
      torch.equal(post_construction[0], ref42_plus_model_build_torch))
print("    post-construction == fresh seed 43 (negative control) :", states_equal(post_construction, ref43))
print("    pre-resume perturbed == fresh seed 42                  :", states_equal(pre_resume, ref42))
print("    after resume surrogate == fresh seed 42               :", states_equal(after_resume, ref42))
````

## Appendix B — `b58_ch3_search.py`

Ran unmodified on the host in T7, against `docs/reference/ch3.pdf` only.

sha256 `e29cbd8672690d250f27ae72c65c9f3e6ff493ae4a44f65a1bf76926d9e7a412` · 47 lines · 2054 bytes

````python
"""Read-only text search of the REPO copy of Chapter III (docs/reference/ch3.pdf only). Raw values.

This PDF's text layer separates tokens with U+200B ZERO WIDTH SPACE, which Python's \\s does not match,
and table cells split words mid-token. Each page is normalised by deleting zero-width characters,
joining end-of-line hyphenation and collapsing whitespace; each needle is matched twice: normally, and
with all whitespace removed from both sides. Positive control: CUBLAS_WORKSPACE_CONFIG on the p.121 row.
"""
import hashlib
import re

try:
    import pymupdf as fitz
except ImportError:  # older PyMuPDF exposes only the fitz name
    import fitz

PATH = "docs/reference/ch3.pdf"
ZW = re.compile(r"[​‌‍﻿]")
NEEDLES = ["CUBLAS_WORKSPACE_CONFIG", "will also remain the same", "seed 42",
           "before CUDA initialization where applicable"]


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"-\n(?=[a-z])", "", ZW.sub("", s)))


data = open(PATH, "rb").read()
print("file", PATH, "| bytes", len(data), "| sha256", hashlib.sha256(data).hexdigest())
doc = fitz.open(PATH)
print("pdf pages", doc.page_count)
pages = [(i, page.get_text()) for i, page in enumerate(doc, 1)]
for needle in NEEDLES:
    spaced, squeezed = 0, 0
    for i, raw in pages:
        flat = norm(raw)
        nospace = re.sub(r"\s+", "", flat).lower()
        s_hits = list(re.finditer(re.escape(needle), flat, flags=re.I))
        q_hits = nospace.count(re.sub(r"\s+", "", needle).lower())
        spaced += len(s_hits)
        squeezed += q_hits
        if s_hits or q_hits:
            lines = [ZW.sub("", ln).strip() for ln in raw.splitlines() if ZW.sub("", ln).strip()]
            print(f"[{needle}] pdf page {i} | spaced hits {len(s_hits)} | whitespace-free hits {q_hits} "
                  f"| last line {lines[-1]!r}")
            for m in s_hits:
                lo, hi = max(0, m.start() - 170), min(len(flat), m.end() + 70)
                print(f"    ...{flat[lo:hi]}...")
    print(f"[{needle}] totals: spaced {spaced} | whitespace-free {squeezed}")
````
