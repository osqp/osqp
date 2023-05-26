/* OSQP TESTER MODULE */

#define CATCH_CONFIG_RUNNER
#include <catch2/catch.hpp>

/* All testcases are in their own cpp file in the subdirectories.
   This file only defines the main function for CATCH to use.
 */

#include <cstdlib>
#include <iostream>

#include "osqp_tester.h"

int OSQPTestFixture::deviceNumber = 0;

int main( int argc, char* argv[] ) {
    Catch::Session session;

    int result = 0;
    int deviceNum = 0;

    /*
     * Select the device to run the test on.
     */

    // Start by looking for the environment variable
    if(const char* deviceEnv = std::getenv("OSQP_TEST_DEVICE_NUM")) {
        OSQPTestFixture::deviceNumber = std::atoi(deviceEnv);
    }

    using namespace Catch::clara;

    // Next use a command line option (if it exists)
    auto cli = session.cli() | Opt(OSQPTestFixture::deviceNumber, "device")
                                  ["--device"]
                                  ("Compute device (overrides the OSQP_TEST_DEVICE_NUM environment variable)");
    session.cli(cli);

    result = session.applyCommandLine( argc, argv );

    if( result != 0 ) {
        // Indicates a command line error
        std::cout << "Error configuring command line for test driver" << std::endl;
        return result;
    }

    /*
     * Run the OSQP test suite
     */
    return session.run( argc, argv );
}
