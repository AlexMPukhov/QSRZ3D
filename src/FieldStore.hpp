// Layout of the (xi, r, component) arrays that hold the fields seen by the
// beams and the beam sources seen by the plasma.  Components of mode 1 are
// only present when azimuthal mode 1 is enabled.
#pragma once

namespace quarz {

// fields for the beam push:  E+ = E_x + i E_y and B+ in V+ modes, E_z and B_z in scalar modes
enum FieldComp : int {
    F_EP1R = 0, F_EP1I, F_BP1R, F_BP1I, F_EZ0, F_BZ0,                    // mode 0
    F_EP0R, F_EP0I, F_EP2R, F_EP2I, F_BP0R, F_BP0I, F_BP2R, F_BP2I,      // mode 1 (vectors)
    F_EZ1R, F_EZ1I, F_BZ1R, F_BZ1I,                                      // mode 1 (scalars)
    F_NC0 = 6, F_NC1 = 18
};

// beam sources: rho - J_z, J_z, J+, rho
enum BeamComp : int {
    B_RT0 = 0, B_JZ0, B_JP1R, B_JP1I, B_RHO0,
    B_RT1R, B_RT1I, B_JZ1R, B_JZ1I, B_JP0R, B_JP0I, B_JP2R, B_JP2I, B_RHO1R, B_RHO1I,
    B_NC0 = 5, B_NC1 = 15
};

} // namespace quarz
