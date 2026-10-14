#version 450

layout(push_constant) uniform DrawConstants {
   vec4 color;
   vec4 params;
} pc;

void main()
{
   vec2 position = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
   gl_Position = vec4(position * 2.0 - 1.0, pc.params.x, 1.0);
}
