import nimblephysics as nimble

# Load the model
rajagopal_opensim: nimble.biomechanics.OpenSimFile = nimble.RajagopalHumanBodyModel()
skeleton: nimble.dynamics.Skeleton = rajagopal_opensim.skeleton

# Create a GUI
gui = nimble.NimbleGUI()

# Serve the GUI on port 8080
gui.serve(8888)

# Render the skeleton to the GUI
gui.nativeAPI().renderSkeleton(skeleton)

# Block until the GUI is closed
gui.blockWhileServing()