#include "etiss/IntegratedLibrary/MemMappedPeriph.h"
#include "etiss/Plugin.h"
#include "etiss/CPUArch.h"

#include <stdio.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdlib.h>
#include <errno.h>

#ifndef ETISS_PLUGIN_QVANILLAACCELERATOR_H
#define ETISS_PLUGIN_QVANILLAACCELERATOR_H

namespace etiss
{
namespace plugin
{
class QVanillaAccelerator: public etiss::plugin::MemMappedPeriph, public etiss::CoroutinePlugin
{
public:
    void write32(uint64_t addr, int32_t val);
    int32_t read32(uint64_t addr);
    etiss::int32 execute();
    etiss::int32 executionEnd();
    
    MappedMemory getMappedMem() const {
        MappedMemory mm;
        mm.base = 0x70000000;
        mm.size = 0x34;
        return mm;
    }

private:
    struct RegIF
    {
        uint32_t ifmap;   
        uint32_t weights;
        uint32_t bias; 
        uint32_t result;  
        int32_t oc;      
        int32_t iw;      
        int32_t ih;      
        int32_t ic;      
        int32_t kh;      
        int32_t kw; 
        int32_t i_zp;
        int32_t k_zp;
        int32_t control;
        int32_t status;
    };

    RegIF regIf;
    etiss::uint64 old_cycles_ = 0;
    etiss::uint64 target_time = 1000000; // 

    // Newly added variables for tracking convolution computation
    bool isComputing = false;            // To track if convolution computation is ongoing
    etiss::uint64 startCycles = 0;       // To record the cycles at the start of computation
    etiss::int64 post_computation_cycles = 0;
    etiss::uint64 start_time_;
     etiss::uint64 time_elapsed;
protected:
    std::string _getPluginName() const;
};

} // namespace plugin
} // namespace etiss

#endif // ETISS_PLUGIN_QVANILLAACCELERATOR_H