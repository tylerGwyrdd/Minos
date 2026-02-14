# Minos
Minos is a parafoil simulation and guidance research project built around a
modular architecture:

- `minos.physics` for 6-DoF plant dynamics,
- `minos.gnc` for navigation, guidance, and control strategies,
- `minos.sim.runners` for canonical open-loop and closed-loop execution loops,
- `minos.identification` for aerodynamic coefficient fitting workflows.
- `minos.parallel` for reusable sequential/thread/process task execution.

The project is organized to support repeatable strategy comparison:

1. Keep one shared physics model.
2. Swap GnC components via interfaces.
3. Run both through the same simulation runner layer.
4. Compare behavior using common snapshot logging and plotting utilities.

If you are new to the repo, start with:

1. `usage.md` for setup and first runs.
2. `architecture.md` for how modules fit together.
3. `examples.md` for runnable scenarios.

## Why?

You might ask why I did this project and I would tell you there are two reasons. Firstly, and the problem aimed to be solved, model rockets very often like to drift with the wind once the parachute opens. The pesky things then drift for miles and often end up in trees so what a better way to prevent this then just swap the simple parachute for a super complicated control system and tangle loving parafoil. Secondly, and the reason I ended up doing this physics/areodynamicist/compsci project as a electronic student was low and behold, no one else in the team wanted to do simulation.

Apparently you cannot do any acidemic engineering projects without simulation? That and we thought there wouldn't be enough work for 5 students (there definitely was) so the simulation can of worms was opened and politely handed to me. Anyways, I digress. I knew right from the start that this would be an absolute ball ache and I was not wrong.

### My take on Minos

Overall though, a super fun project and I was happy to fit in some genetic algorithmns in for model Identifion. This was by far the most interesting part, even tho most of that work was done 3 days before the deadline. Guidance is very interesting and there is soooo much more scope then I did for minos (dont copy Minos guidance, it is subpar).

Physics models are hard. Coding them in are just as bad. Realistically, the work in Minos is a generic 6D0F model, it doesnt at all consider the rocket and its long pedulmn shape. Not at all realistic for tuning control algorithmns which is what the aim of Minos was.

If you enjoy life (well I guess you are reading this so clearly there are some doubts), stick to normal parachutes and go launch some rockets. check out LURA.

### Tyler, please can I read your dissatation?

Nice try good sir, fortuantely there are books and papers which do a far better job  at everything then the 30 page sleepless mess that I unconsiously produced during a blur of academic overloading. If I was good boy in finishing this documentation, I will have copied over some parts for explinations for specific features or concepts. Instead I will list some stupendus references that were odanied bibles during the project tenure. (TLDR No feck off, read shite)
