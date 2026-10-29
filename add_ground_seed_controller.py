"""
Adds a 'Terrain Seed Displace' procedural Geometry Nodes modifier to 'Forest Generator'.

Specifications:
- Ground Seed: Integer only (NodeSocketInt)
- Ground Height: Limited strictly to [-1.0, 1.0]
- Ground Scale: Locked internally at 0.030 (no modifier slider)
"""
import bpy


def setup_terrain_seed():
    fg = bpy.data.objects.get("Forest Generator")
    if not fg:
        print("Forest Generator not found!")
        return

    # 1. Clean up old node group if it exists
    old_ng = bpy.data.node_groups.get("Terrain Ground Randomizer")
    if old_ng:
        bpy.data.node_groups.remove(old_ng, do_unlink=True)

    # 2. Create clean node group
    ng = bpy.data.node_groups.new(name="Terrain Ground Randomizer", type='GeometryNodeTree')

    # Interface sockets
    ng.interface.new_socket('Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')

    # Ground Seed: INT only
    s_seed = ng.interface.new_socket('Ground Seed', in_out='INPUT', socket_type='NodeSocketInt')
    s_seed.default_value = 0
    s_seed.min_value = 0

    # Ground Height: Float limited between -1.0 and 1.0
    s_height = ng.interface.new_socket('Ground Height', in_out='INPUT', socket_type='NodeSocketFloat')
    s_height.default_value = 0.5
    s_height.min_value = -1.0
    s_height.max_value = 1.0

    # Output Geometry
    ng.interface.new_socket('Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')

    # 3. Construct Node Graph
    in_node = ng.nodes.new('NodeGroupInput')
    out_node = ng.nodes.new('NodeGroupOutput')
    set_pos = ng.nodes.new('GeometryNodeSetPosition')

    # 4D Noise Texture (Seed controls W)
    noise = ng.nodes.new('ShaderNodeTexNoise')
    noise.noise_dimensions = '4D'
    # Lock scale strictly at 0.030 (no socket exposed)
    noise.inputs['Scale'].default_value = 0.030
    noise.inputs['Detail'].default_value = 3.0
    noise.inputs['Roughness'].default_value = 0.5

    # Center noise around 0 (Subtract 0.5)
    sub = ng.nodes.new('ShaderNodeMath')
    sub.operation = 'SUBTRACT'
    sub.inputs[1].default_value = 0.5

    # Multiply by Ground Height
    mul = ng.nodes.new('ShaderNodeMath')
    mul.operation = 'MULTIPLY'

    # Combine into Z displacement
    comb = ng.nodes.new('ShaderNodeCombineXYZ')

    # Wire up nodes
    ng.links.new(in_node.outputs['Geometry'], set_pos.inputs['Geometry'])
    # Connect integer seed to float W input
    ng.links.new(in_node.outputs['Ground Seed'], noise.inputs['W'])
    ng.links.new(noise.outputs['Fac'], sub.inputs[0])
    ng.links.new(sub.outputs['Value'], mul.inputs[0])
    ng.links.new(in_node.outputs['Ground Height'], mul.inputs[1])
    ng.links.new(mul.outputs['Value'], comb.inputs['Z'])
    ng.links.new(comb.outputs['Vector'], set_pos.inputs['Offset'])
    ng.links.new(set_pos.outputs['Geometry'], out_node.inputs['Geometry'])

    # 4. Attach modifier to Forest Generator
    mod = fg.modifiers.get("Terrain Seed Displace")
    if not mod:
        mod = fg.modifiers.new(name="Terrain Seed Displace", type='NODES')

    mod.node_group = ng

    # Ensure modifier is at top of stack (before tree scatter)
    fg.modifiers.move(fg.modifiers.find("Terrain Seed Displace"), 0)

    bpy.context.view_layer.update()
    print("Terrain Seed Displace updated successfully!")
    print("- Ground Seed: Integer only")
    print("- Ground Height: Range [-1.0, 1.0]")
    print("- Ground Scale: Locked at 0.030")


if __name__ == "__main__":
    setup_terrain_seed()
