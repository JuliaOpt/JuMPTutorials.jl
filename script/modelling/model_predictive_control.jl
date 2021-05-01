#' ---
#' title: Model-Predictive Control
#' ---

#' **Originally Contributed by**: Robin Deits

#' Let's implement a very simple model-predictive control (MPC) optimization in JuMP. 
#' Specifically, we'll write an optimization that tries to find a sequence of control inputs 
#' for a very simple model robot in order to optimize an objective function. 

#' More concretely, our model will be a brick, sliding frictionlessly in two dimensions. 
#' Our input will be the acceleration of the brick, and we'll try to minimize the brick's final velocity and distance from the origin. 

using JuMP, Ipopt

model = Model(with_optimizer(Ipopt.Optimizer, print_level=0))

# Define our constant parameters
Δt = 0.1
num_time_steps = 20
max_acceleration = 0.5

# Define our decision variables
@variables(model, begin
    position[1:2, 1:num_time_steps]
    velocity[1:2, 1:num_time_steps]
    -max_acceleration <= acceleration[1:2, 1:num_time_steps] <= max_acceleration
end)

# Add dynamics constraints
@constraint(model, [i=2:num_time_steps, j=1:2],
            velocity[j, i] == velocity[j, i - 1] + acceleration[j, i - 1] * Δt)
@constraint(model, [i=2:num_time_steps, j=1:2],
            position[j, i] == position[j, i - 1] + velocity[j, i - 1] * Δt)

# Cost function: minimize final position and final velocity
@objective(model, Min, 
    100 * sum(position[:, end].^2) + sum(velocity[:, end].^2))

# Initial conditions:
@constraint(model, position[:, 1] .== [1, 0])
@constraint(model, velocity[:, 1] .== [0, -1])

optimize!(model)

#+ 

# Extract the solution from the model
q = value.(position)
v = value.(velocity)
u = value.(acceleration)

#' ### Drawing the Result 
#' We can draw the output of the optimization using the `Plots.jl` package

using Plots
# Use the GR backend for Plots.jl, because it's fast
gr()

#+ 

# The @animate macro creates an animated plot, which lets us draw the
# optimized trajectory of the brick as a function of time
anim = @animate for i = 1:num_time_steps
    plot(q[1, :], q[2, :], xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
    plot!([q[1, i]], [q[2, i]], marker=(:hex, 6))
end

#+

# The gif() function saves our animated plot to an animated gif
# Note: this may require you to have the `ffmpeg` program installed.
# On Ubuntu 16.04, you can get this with `sudo apt-get install ffmpeg`.
gif(anim, "img/mpc1.gif", fps = 30)

#' ## Running the MPC Controller 
#' In a real application, we wouldn't just run the MPC optimization once. 
#' Instead, we might run the optimization at every time step using the robot's current state. 
#' To do that, let's wrap the MPC problem in a function called `run_mpc()` that takes the robot's current position and velocity as input:

# run_mpc() takes the robot's current position and velocity
# and returns an optimized trajectory of position, velocity, 
# and acceleration. 
function run_mpc(initial_position, initial_velocity)
    
    model = Model(with_optimizer(Ipopt.Optimizer, print_level=0))

    Δt = 0.1
    num_time_steps = 10
    max_acceleration = 0.5

    @variables(model, begin
        position[1:2, 1:num_time_steps]
        velocity[1:2, 1:num_time_steps]
        -max_acceleration <= acceleration[1:2, 1:num_time_steps] <= max_acceleration
    end)

    # Dynamics constraints
    @constraint(model, [i=2:num_time_steps, j=1:2],
                velocity[j, i] == velocity[j, i - 1] + acceleration[j, i - 1] * Δt)
    @constraint(model, [i=2:num_time_steps, j=1:2],
                position[j, i] == position[j, i - 1] + velocity[j, i - 1] * Δt)

    # Cost function: minimize final position and final velocity
    @objective(model, Min, 
        100 * sum(position[:, end].^2) + sum(velocity[:, end].^2))

    # Initial conditions:
    @constraint(model, position[:, 1] .== initial_position)
    @constraint(model, velocity[:, 1] .== initial_velocity)

    optimize!(model)
    return value.(position), value.(velocity), value.(acceleration)
end

#' We can demonstrate this by repeatedly running the MPC program, applying its planned acceleration to the brick, 
#' and then simulating one step forward in time: 

# The robot's starting position and velocity
q = [1.0, 0.0]
v = [0.0, -1.0]

anim = @animate for i in 1:80
    # Plot the current position
    plot([q[1]], [q[2]], marker = (:hex, 10), xlim = (-1.1, 1.1), ylim = (-1.1, 1.1))
    
    # Run the MPC control optimization
    q_plan, v_plan, u_plan = run_mpc(q, v)
    
    # Draw the planned future states from the MPC optimization
    plot!(q_plan[1, :], q_plan[2, :], linewidth=5)
    
    # Apply the planned acceleration and simulate one step in time
    u = u_plan[:, 1]
    v += u * Δt
    q += v * Δt
end

#+

gif(anim, "img/mpc2.gif") 