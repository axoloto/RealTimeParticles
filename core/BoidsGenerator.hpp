
#pragma once 

namespace Core {
    class BoidsGenerator {
        public:
            BoidsGenerator(size_t id);
            ~BoidsGenerator() = default;
        private:
            size_t m_id;
    };
}

