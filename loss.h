#pragma once

#include "node.h"


ExecNode *loss_create_cross_entropy_with_logits(Arena *arena, usz *counter, ExecNode *logits, ExecNode *target);
