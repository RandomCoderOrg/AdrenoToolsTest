#version 450

layout(push_constant) uniform DrawConstants {
   vec4 color;
   vec4 params;
} pc;

layout(location = 0) out vec4 color;

void main()
{
   color = pc.color;
}
