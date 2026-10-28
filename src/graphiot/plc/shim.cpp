// Generic C ABI around a STruC++-generated PLC program (POU `Main`, config
// `Config0`). Compiled together with the generated `generated.cpp`, so it can
// reference strucpp::Program_MAIN and the file-scope strucpp::locatedVars[].
//
// The program writes its outputs straight into its own member storage each
// scan; strucpp::locatedVars[] only *binds* that storage to I/O addresses, and
// it is rebound to whichever Program_MAIN was most recently constructed. So we
// snapshot the bound pointers into a per-handle table right after construction
// (construction is serialised by the caller's GIL). Multiple instances of the
// same program — the live soft-PLC and the oracle twin's counterfactual replay
// — can then coexist in one process, each reading/writing its own storage.
#include "generated.hpp"
#include <cstdint>
#include <cstring>

using namespace strucpp;

namespace {

struct Handle {
    Program_MAIN *prog;
    int64_t t_ns;
    int n;
    LocatedVar *vars;  // per-instance snapshot of the bound located variables
};

LocatedVar *find(Handle *h, int area, int byte_index, int bit_index, int is_bit) {
    for (int i = 0; i < h->n; ++i) {
        LocatedVar &v = h->vars[i];
        int vbit = (v.size == LocatedSize::Bit) ? 1 : 0;
        if (static_cast<int>(v.area) == area && vbit == is_bit &&
            v.byte_index == byte_index &&
            (!vbit || v.bit_index == bit_index)) {
            return &v;
        }
    }
    return nullptr;
}

}  // namespace

extern "C" {

void *plc_new(void) {
    Handle *h = new Handle();
    h->prog = new Program_MAIN();  // constructor rebinds the global locatedVars[]
    h->t_ns = 0;
    h->n = static_cast<int>(locatedVarsCount);
    h->vars = new LocatedVar[h->n];
    for (int i = 0; i < h->n; ++i) h->vars[i] = locatedVars[i];  // snapshot this instance
    __CURRENT_TIME_NS = 0;
    return h;
}

void plc_free(void *hv) {
    Handle *h = static_cast<Handle *>(hv);
    delete[] h->vars;
    delete h->prog;
    delete h;
}

// Advance simulated time and run exactly one scan.
void plc_step(void *hv, int64_t dt_ns) {
    Handle *h = static_cast<Handle *>(hv);
    h->t_ns += dt_ns;
    __CURRENT_TIME_NS = h->t_ns;
    h->prog->run();
}

int plc_count(void *hv) { return static_cast<Handle *>(hv)->n; }

// Snapshot / restore of the whole program state (timers, phase, every variable)
// for the oracle's counterfactual replays. A plain byte copy is valid because it
// is always between objects of the same type in the same process (same vtable)
// and the program holds no pointers to itself.
int plc_state_size(void) { return static_cast<int>(sizeof(Program_MAIN)); }

int64_t plc_save(void *hv, void *buf) {
    Handle *h = static_cast<Handle *>(hv);
    std::memcpy(buf, static_cast<void *>(h->prog), sizeof(Program_MAIN));
    return h->t_ns;
}

void plc_restore(void *hv, const void *buf, int64_t t_ns) {
    Handle *h = static_cast<Handle *>(hv);
    std::memcpy(static_cast<void *>(h->prog), buf, sizeof(Program_MAIN));
    h->t_ns = t_ns;
}

// Expose the i-th located variable's address descriptor (for introspection/tests).
void plc_meta(void *hv, int i, int *area, int *size, int *byte_index, int *bit_index) {
    LocatedVar &v = static_cast<Handle *>(hv)->vars[i];
    *area = static_cast<int>(v.area);
    *size = static_cast<int>(v.size);
    *byte_index = v.byte_index;
    *bit_index = v.bit_index;
}

// area: 0 Input(%I), 1 Output(%Q), 2 Memory(%M). is_bit: 1 for %*X, 0 for %*W.
int32_t plc_read(void *hv, int area, int byte_index, int bit_index, int is_bit) {
    LocatedVar *v = find(static_cast<Handle *>(hv), area, byte_index, bit_index, is_bit);
    if (!v || !v->pointer) return 0;
    if (v->size == LocatedSize::Bit) return *static_cast<uint8_t *>(v->pointer) ? 1 : 0;
    return *static_cast<int16_t *>(v->pointer);
}

int plc_write(void *hv, int area, int byte_index, int bit_index, int is_bit, int32_t value) {
    LocatedVar *v = find(static_cast<Handle *>(hv), area, byte_index, bit_index, is_bit);
    if (!v || !v->pointer) return 0;
    if (v->size == LocatedSize::Bit)
        *static_cast<uint8_t *>(v->pointer) = value ? 1 : 0;
    else
        *static_cast<int16_t *>(v->pointer) = static_cast<int16_t>(value);
    return 1;
}

}  // extern "C"
