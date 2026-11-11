"""PPO rewards: terminal +/-1, tower-hit-point potential, elixir-overflow penalties and the King-activation penalty.

Canonical match results stay +/-1/0. The tower calculator is BattleEnvV1's own, reached through a lightweight reward state,
so the native environment's rewards are not read twice.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Mapping

sys.path[:0] = [str(Path(__file__).resolve().parents[1])]

from native_runner.battle_env import BattleEnvV1, _elixir_generated_raw   # noqa: E402
from native_runner.cr_native_env import COMMAND_EXECUTION_DELAY_TICKS   # noqa: E402
from native_runner.match_factory import NATIVE_GAMEPLAY_END_TICK   # noqa: E402
from native_runner.training.v4.king_activation_reward import apply_king_activation_reward_v4   # noqa: E402

DELAY_CEILING_MS = 200
RECIPE_ID = 'hp02-win1.v1'
REWARD_DEFAULTS = dict(
    terminal_reward_scale=1., reward_gamma=.9997, shaping=.2,
    king_shaping_mode='hp_point_weighted', king_potential_weight=1.,
    personal_elixir_overflow_penalty=.0225, unilateral_elixir_overflow_penalty=.075,
    personal_elixir_overflow_grace=4., elixir_overflow_step_penalty_cap=.1,
    king_activation_penalty=.1, king_activation_scope='any',
)


def reward_recipe(settings=None):
    """The defaults; values present in `settings` override them."""
    settings = settings or {}; recipe = dict(REWARD_DEFAULTS)
    for key, default in list(recipe.items()):
        if key in settings:
            recipe[key] = settings[key] if isinstance(default, str) else float(settings[key])
    recipe.update(schema=RECIPE_ID,
                  shaping_tau_ticks=5. / -math.log(recipe['reward_gamma']), gameplay_end_tick=NATIVE_GAMEPLAY_END_TICK)
    return recipe


def _newly_wasted_elixir_v4(*, before_elixir: float, after_elixir: float, generated_elixir: float) -> float:
    """Exact newly discarded elixir for one observed engine transition."""
    if any(not math.isfinite(value) or value < 0.0 for value in (before_elixir, after_elixir, generated_elixir)):
        raise ValueError("elixir telemetry inputs must be finite and nonnegative")
    if after_elixir < 10.0 - 1e-6:
        return 0.0
    return generated_elixir if before_elixir >= 10.0 - 1e-6 else max(0.0, before_elixir + generated_elixir - 10.0)


def _scaled_terminal_rewards_v4(reward: Mapping[int, float], terminal: Any, scale: float) -> dict[int, float]:
    """Scale only the terminal outcome; leave potential shaping untouched."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("terminal reward scale must be finite and positive")
    return {
        owner: float(reward[owner]) + ((scale - 1.) * float(terminal.result_by_owner[owner])
                                     if terminal.ended else 0.)
        for owner in (0, 1)
    }


def _elixir_overflow_penalty_transition_v4(
    *,
    personal_wasted_elixir: float,
    personal_accrued_cost: float,
    unilateral_wasted_elixir: float,
    unilateral_accrued_cost: float,
    before_elixir: float,
    after_elixir: float,
    generated_elixir: float,
    both_full: bool,
    personal_coefficient: float,
    personal_grace_elixir: float,
    unilateral_coefficient: float,
    step_penalty_cap: float,
) -> tuple[float, float, float, float, float, float]:
    """Advance independent personal-overflow and unilateral-overflow clocks."""

    values = (
        personal_wasted_elixir,
        personal_accrued_cost,
        unilateral_wasted_elixir,
        unilateral_accrued_cost,
        before_elixir,
        after_elixir,
        generated_elixir,
        personal_coefficient,
        personal_grace_elixir,
        unilateral_coefficient,
        step_penalty_cap,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError("elixir overflow penalty inputs must be finite and nonnegative")
    if step_penalty_cap <= 0.0:
        raise ValueError("elixir overflow step penalty cap must be positive")
    if after_elixir < 10.0 - 1e-6:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    newly_wasted = _newly_wasted_elixir_v4(
        before_elixir=before_elixir, after_elixir=after_elixir, generated_elixir=generated_elixir
    )
    next_personal_wasted = personal_wasted_elixir + newly_wasted
    personal_target_cost = personal_coefficient * max(0.0, next_personal_wasted - personal_grace_elixir)
    personal_delta = max(0.0, personal_target_cost - personal_accrued_cost)
    if both_full:
        next_unilateral_wasted = 0.0
        unilateral_accrued_cost = 0.0
        unilateral_delta = 0.0
    else:
        next_unilateral_wasted = unilateral_wasted_elixir + newly_wasted
        unilateral_target_cost = unilateral_coefficient * next_unilateral_wasted * next_unilateral_wasted
        unilateral_delta = max(0.0, unilateral_target_cost - unilateral_accrued_cost)
    combined_delta = personal_delta + unilateral_delta
    scale = 1.0 if combined_delta <= step_penalty_cap else step_penalty_cap / combined_delta
    personal_penalty = personal_delta * scale
    unilateral_penalty = unilateral_delta * scale
    return (
        next_personal_wasted,
        personal_accrued_cost + personal_penalty,
        next_unilateral_wasted,
        unilateral_accrued_cost + unilateral_penalty,
        personal_penalty,
        unilateral_penalty,
    )


def deadline_cancellation(event):
    """A fully funded/legal touch queued before gameplay ended but arriving after.

    Keep the raw rejection in diagnostics. These are expected deadline expiries,
    rather than permitting post-horizon policy actions or hiding active rejects.
    """
    kind=event.event_type if hasattr(event,'event_type') else event.get('event_type')
    d=event.data if hasattr(event,'data') else event.get('data',{})
    diag=d.get('rejection_diagnostic',{})
    try:
        requested=int(d['requested_tick']);expected=int(d['expected_execution_tick'])
        cost=float(d['effective_cost'])*10000
        before=float(diag['before_elixir_raw']);after=float(diag['after_elixir_raw'])
        # Delay-aware legality can select a card just before it becomes funded.
        # The observed receipt includes the fixed server execution delay; its
        # validated touch occurs earlier. Keep the allowance bounded to the
        # configured 200ms ceiling + one touch tick + 0..3 training jitter.
        touch=expected-COMMAND_EXECUTION_DELAY_TICKS
        max_touch_ticks=DELAY_CEILING_MS//50+1+3
        touch_funded=(0<=touch-requested<=max_touch_ticks and requested<NATIVE_GAMEPLAY_END_TICK
            and before+_elixir_generated_raw(requested,min(touch,NATIVE_GAMEPLAY_END_TICK))>=cost)
        funded=(before>=cost or touch_funded) and after>=cost
    except (KeyError,ValueError,TypeError):return False
    return bool(kind=='action_rejected' and d.get('kind')=='play_card'
        and requested<NATIVE_GAMEPLAY_END_TICK<=expected and funded
        and diag.get('mask_target_legal') is True and diag.get('same_card_still_in_slot') is True
        and diag.get('resolution_mode')=='resident_hand_slot_attestation'
        and diag.get('native_card_play_candidates') in ([],()))


def _pair(observations):
    return {0:observations,1:observations} if hasattr(observations,'towers') else observations


class RewardAdapter:
    """Episode-scoped reward history, independent of actor state.

    transition(before,next,policy_active) returns (rewards,replace_last).
    Normal policy transitions append rewards. The gameplay-boundary transition
    initially appends zero; its overflow/activation costs are frozen at tick
    6000. Later native settlement emits no rows and replaces that final reward
    once the canonical terminal arrives. Nonterminal settlement returns None.
    """
    def __init__(self,environment,initial_observations,settings=None):
        self.recipe=reward_recipe(settings);self.environment=environment
        initial=_pair(initial_observations);towers=initial[0].towers
        # Only the reward methods are used: a full second BattleEnv would load catalogs or touch a native session.
        calculator=object.__new__(BattleEnvV1)
        calculator.shaping_beta=self.recipe['shaping'];calculator.shaping_tau_ticks=self.recipe['shaping_tau_ticks']
        calculator.king_shaping_mode=self.recipe['king_shaping_mode'];calculator.king_potential_weight=self.recipe['king_potential_weight']
        known=getattr(environment,'_initial_tower_hp',{})
        calculator._initial_tower_hp={tower.entity_id:float(known.get(tower.entity_id,tower.max_hitpoints)) for tower in towers}
        calculator._king_loss_baseline={0:0.,1:0.};calculator._update_king_loss_baseline(towers)
        calculator._last_potential={owner:calculator._potential(owner,towers) for owner in (0,1)}
        self.calculator=calculator;self.pending=None;self.finished=False
        self.personal_wasted=[0.,0.];self.personal_accrued=[0.,0.]
        self.unilateral_wasted=[0.,0.];self.unilateral_accrued=[0.,0.]
        self.personal_penalty_total=[0.,0.];self.unilateral_penalty_total=[0.,0.]
        self.shaping_total=[0.,0.];self.terminal_total=[0.,0.]
        self.activation=SimpleNamespace(environment=environment,king_activation_seen_owners=set(),
            king_activations=(0,0),king_activation_tick=(None,None),king_activation_source_card_id=(None,None),
            king_activation_source_unattributed=(0,0),king_activation_reward_total=(0.,0.),king_activation_details=[])

    def _base_reward(self,before,after):
        terminal=after[0].terminal
        base=self.calculator._rewards(before[0].towers,after[0].towers,terminal,
            max(0,int(after[0].tick)-int(before[0].tick)))
        for owner in (0,1):
            outcome=float(terminal.result_by_owner[owner]) if terminal.ended else 0.
            self.shaping_total[owner]+=base[owner]-outcome
            self.terminal_total[owner]+=self.recipe['terminal_reward_scale']*outcome
        # _rewards supplies canonical +/-1 exactly once; this helper adds only
        # (scale-1)*result, matching the last old PPO collector.
        return _scaled_terminal_rewards_v4(base,terminal,self.recipe['terminal_reward_scale'])

    def _active_adjustments(self,before,after):
        start=int(before[0].tick);stop=min(int(after[0].tick),NATIVE_GAMEPLAY_END_TICK)
        generated=_elixir_generated_raw(start,stop)/10000.
        before_elixir=[float(next(p for p in before[owner].players if p.owner==owner).elixir_exact) for owner in (0,1)]
        after_elixir=[float(next(p for p in after[owner].players if p.owner==owner).elixir_exact) for owner in (0,1)]
        both_full=all(value>=10.-1e-6 for value in after_elixir);adjusted={0:0.,1:0.}
        for owner in (0,1):
            values=_elixir_overflow_penalty_transition_v4(
                personal_wasted_elixir=self.personal_wasted[owner],personal_accrued_cost=self.personal_accrued[owner],
                unilateral_wasted_elixir=self.unilateral_wasted[owner],unilateral_accrued_cost=self.unilateral_accrued[owner],
                before_elixir=before_elixir[owner],after_elixir=after_elixir[owner],generated_elixir=generated,
                both_full=both_full,personal_coefficient=self.recipe['personal_elixir_overflow_penalty'],
                personal_grace_elixir=self.recipe['personal_elixir_overflow_grace'],
                unilateral_coefficient=self.recipe['unilateral_elixir_overflow_penalty'],
                step_penalty_cap=self.recipe['elixir_overflow_step_penalty_cap'])
            (self.personal_wasted[owner],self.personal_accrued[owner],self.unilateral_wasted[owner],
                self.unilateral_accrued[owner],personal,unilateral)=values
            self.personal_penalty_total[owner]+=personal;self.unilateral_penalty_total[owner]+=unilateral
            adjusted[owner]-=personal+unilateral
        # A terminal native jump beyond gameplay cannot provide a true tick-6000
        # King snapshot. Do not mistake settlement HP subtraction for activation.
        # Normal 5-tick collector decisions capture the boundary before draining.
        if int(after[0].tick)<=NATIVE_GAMEPLAY_END_TICK:
            apply_king_activation_reward_v4(self.activation,before,after,adjusted,
                self.recipe['king_activation_penalty'],self.recipe['king_activation_scope'])
        return adjusted

    def transition(self,before_observations,next_observations,policy_active):
        if self.finished:raise RuntimeError('terminal reward already emitted')
        before=_pair(before_observations);after=_pair(next_observations)
        if policy_active:
            if self.pending is not None or int(before[0].tick)>=NATIVE_GAMEPLAY_END_TICK:
                raise RuntimeError('policy transition after gameplay end')
            extra=self._active_adjustments(before,after)
            if int(after[0].tick)>=NATIVE_GAMEPLAY_END_TICK and not after[0].terminal.ended:
                if int(after[0].tick)!=NATIVE_GAMEPLAY_END_TICK:
                    raise RuntimeError('native drain must capture the gameplay boundary before settlement')
                self.pending=(before,extra)
                return {0:0.,1:0.},False
            values=self._base_reward(before,after)
            values={owner:values[owner]+extra[owner] for owner in (0,1)}
            self.finished=bool(after[0].terminal.ended)
            return values,False
        if not after[0].terminal.ended:return None,False
        if self.pending is None:raise RuntimeError('settlement has no preceding real policy transition')
        last_before,extra=self.pending;values=self._base_reward(last_before,after)
        self.pending=None;self.finished=True
        return {owner:values[owner]+extra[owner] for owner in (0,1)},True

    def report(self):
        return dict(reward_recipe=self.recipe,terminal_outcome_reward=tuple(self.terminal_total),
            tower_shaping_reward=tuple(self.shaping_total),personal_elixir_overflow_penalty_total=tuple(self.personal_penalty_total),
            unilateral_elixir_overflow_penalty_total=tuple(self.unilateral_penalty_total),
            king_activation_reward_total=self.activation.king_activation_reward_total,
            king_activation_details=list(self.activation.king_activation_details))
