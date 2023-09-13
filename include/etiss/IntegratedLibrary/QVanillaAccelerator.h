#include "etiss/IntegratedLibrary/MemMappedPeriph.h"
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>
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
     
        MappedMemory getMappedMem() const {
            MappedMemory mm;
            mm.base = 0x70000000;
            mm.size = 0x38;
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
        static const int64_t cycles_per_mac = 4; 
        etiss::uint64 target_time ; 
        etiss::uint64 start_time_ ;
        int64_t num_macs = 0; 
        bool myflag = false;
        bool myflag2 = false;
        int64_t count = 0;
        int32_t computation_status = 0; 
    protected:
        std::string _getPluginName() const;

};

} // namespace plugin

} // namespace etiss

#endif