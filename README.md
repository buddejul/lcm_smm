# SMM for LCM Topics Class

Simulated methods of moment for LCM topics class.

Pretend that your OLS model
$Y = X\beta + \epsilon$ does not have an analytical solution and that you just know the basic moments:
$E[X'Y]$.

Simulate the model under this DGP, using a distribution of your choice for $\epsilon$.
Plug the difference between the simulated moments and the observed moments into a criterion function and minimize it.

Use `estimagic` for this task.
