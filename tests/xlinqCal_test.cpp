/** \file xlinqCal_test.cpp
 * \brief Unit and simulator integration tests for xlinqCal.
 */

#include "catch.hpp"

#include <filesystem>
#include <unistd.h>

#include "xlinqCal.hpp"
#include "xlinqSim.hpp"

/// \cond
namespace xlinqCalTest
{

/// A unique stream prefix for this test process.
std::string testPrefix()
{
    return "xlsim_t" + std::to_string( getpid() ) + "_";
}

} // namespace xlinqCalTest
/// \endcond

/// Sign sequences alternate +- and -+ per repeat.
TEST_CASE( "signSequence alternates order per repeat", "[xlinqCal]" )
{
    using xlinqCal::xlinqCal;

    REQUIRE( xlinqCal::signSequence( 1 ) == std::vector<int>{ 1, -1 } );
    REQUIRE( xlinqCal::signSequence( 2 ) == std::vector<int>{ 1, -1, -1, 1 } );
    REQUIRE( xlinqCal::signSequence( 3 ) == std::vector<int>{ 1, -1, -1, 1, 1, -1 } );
}

/// Settling is only needed when the sign, and so the DM command, changes.
TEST_CASE( "settleCount counts DM changes", "[xlinqCal]" )
{
    using xlinqCal::xlinqCal;

    REQUIRE( xlinqCal::settleCount( xlinqCal::signSequence( 1 ) ) == 2 );
    REQUIRE( xlinqCal::settleCount( xlinqCal::signSequence( 2 ) ) == 3 );
    REQUIRE( xlinqCal::settleCount( xlinqCal::signSequence( 3 ) ) == 4 );
}

/// Typed copies convert each supported ImageStreamIO type and reject unknown types.
TEST_CASE( "isioStream::copyTyped converts types", "[isioStream]" )
{
    using xlinqCal::isioStream;

    mx::improc::eigenImage<double> im( 2, 2 );

    uint16_t u16[4] = { 1, 2, 3, 65535 };
    isioStream::copyTyped( im, u16, _DATATYPE_UINT16, 2, 2 );
    REQUIRE( im( 0, 0 ) == 1 );
    REQUIRE( im( 1, 0 ) == 2 );
    REQUIRE( im( 0, 1 ) == 3 );
    REQUIRE( im( 1, 1 ) == 65535 );

    int16_t i16[4] = { -1, 2, -3, 4 };
    isioStream::copyTyped( im, i16, _DATATYPE_INT16, 2, 2 );
    REQUIRE( im( 0, 0 ) == -1 );
    REQUIRE( im( 0, 1 ) == -3 );

    float f[4] = { 0.5, 1.5, 2.5, 3.5 };
    isioStream::copyTyped( im, f, _DATATYPE_FLOAT, 2, 2 );
    REQUIRE( im( 1, 1 ) == 3.5 );

    REQUIRE_THROWS( isioStream::copyTyped( im, f, _DATATYPE_HALF, 2, 2 ) );
}

/// isioStream opens a stream, receives a post, and copies the frame with its counter.
TEST_CASE( "isioStream reads a posted frame", "[isioStream]" )
{
    using namespace xlinqCal;

    std::string name = xlinqCalTest::testPrefix() + "isio";

    simStream src;
    src.create( name, _DATATYPE_UINT16, 3, 2 );

    isioStream rd;
    rd.open( name );
    REQUIRE( rd.rows() == 3 );
    REQUIRE( rd.cols() == 2 );
    REQUIRE( rd.semIndex() >= 0 );
    REQUIRE( rd.wait( 0.05 ) == waitResult::timeout );

    for( int i = 0; i < 6; ++i )
    {
        src.data<uint16_t>()[i] = 10 * i;
    }
    src.post();

    REQUIRE( rd.wait( 1.0 ) == waitResult::posted );

    mx::improc::eigenImage<float> im;
    uint64_t                      cnt0;
    REQUIRE( rd.copy( im, cnt0 ) );
    REQUIRE( cnt0 == src.image()->md->cnt0 );
    REQUIRE( im( 2, 0 ) == 20 );
    REQUIRE( im( 0, 1 ) == 30 );

    rd.close();
    REQUIRE_THROWS( rd.open( xlinqCalTest::testPrefix() + "does_not_exist" ) );
}

/// The simulator refuses stream names without its prefix, so it cannot clobber instrument streams.
TEST_CASE( "xlinqSim requires the stream prefix", "[xlinqSim]" )
{
    xlinqCal::xlinqSimConfig cfg;
    cfg.sciName = "camsci1";

    REQUIRE_THROWS_AS( xlinqCal::xlinqSim( cfg ), std::invalid_argument );
}

/// End to end against the simulator: recovered responses match amp * R * shape for both cameras,
/// the timing table is consistent, and the DM channel is restored.
TEST_CASE( "xlinqCal recovers simulated responses", "[xlinqCal][integration]" )
{
    using namespace xlinqCal;

    xlinqSimConfig cfg;
    cfg.prefix       = xlinqCalTest::testPrefix();
    cfg.dmChannel    = cfg.prefix + "dm01disp04";
    cfg.dmComb       = cfg.prefix + "dm01disp";
    cfg.wfsName      = cfg.prefix + "camwfs";
    cfg.sciName      = cfg.prefix + "camsci1";
    cfg.wfsFps       = 1000;
    cfg.sciFps       = 20;
    cfg.sciPostDelay = 0.015;
    cfg.wfsGain      = 100;
    cfg.sciGain      = 200;
    cfg.wfsNoise     = 1;
    cfg.sciNoise     = 1;
    cfg.drift        = 0.01;

    const int   nPoke   = 3;
    const float amp     = 0.1;
    const int   nFrames = 4;

    std::filesystem::path dir = std::filesystem::temp_directory_path() / ( cfg.prefix + "out" );
    std::filesystem::create_directories( dir );

    mx::improc::eigenCube<float> pokes    = xlinqSim::randomPokes( cfg.dmRows, cfg.dmCols, nPoke, 7 );
    std::string                  pokeFile = ( dir / "pokes.fits" ).string();
    mx::fits::fitsFile<float>    ff;
    REQUIRE( ff.write( pokeFile, pokes ) == mx::error_t::noerror );

    xlinqSim sim( cfg );
    sim.start();

    std::vector<std::string> args = { "xlinqCal",
                                      "--dm.channel=" + cfg.dmChannel,
                                      "--dm.comb=" + cfg.dmComb,
                                      "--sci.shmim=" + cfg.sciName,
                                      "--wfs.shmim=" + cfg.wfsName,
                                      "--poke.file=" + pokeFile,
                                      "--poke.amp=" + std::to_string( amp ),
                                      "--sci.nFrames=" + std::to_string( nFrames ),
                                      "--preflight.time=0.5",
                                      "--out.dir=" + dir.string(),
                                      "--out.prefix=test",
                                      "--yes" };

    std::vector<char *> argv;
    for( std::string &a : args )
    {
        argv.push_back( a.data() );
    }

    xlinqCal::xlinqCal app;
    int                rv = app.main( argv.size(), argv.data() );

    REQUIRE( rv == 0 );
    REQUIRE( app.nDone() == nPoke );

    for( int k = 0; k < nPoke; ++k )
    {
        Eigen::Map<const Eigen::VectorXf> shape( pokes.image( k ).data(), cfg.dmRows * cfg.dmCols );

        Eigen::VectorXf sciExp = amp * ( sim.sciR().matrix() * shape );
        Eigen::VectorXf wfsExp = amp * ( sim.wfsR().matrix() * shape );

        Eigen::Map<const Eigen::VectorXf> sciGot( app.sciResp().image( k ).data(), sciExp.size() );
        Eigen::Map<const Eigen::VectorXf> wfsGot( app.wfsResp().image( k ).data(), wfsExp.size() );

        double sciErr = ( sciGot - sciExp ).norm() / sciExp.norm();
        double wfsErr = ( wfsGot - wfsExp ).norm() / wfsExp.norm();

        INFO( "poke " << k << " sci err " << sciErr << " wfs err " << wfsErr );
        REQUIRE( sciErr < 0.05 );
        REQUIRE( wfsErr < 0.05 );
    }

    const std::vector<timingRecord> &timing = app.timing();
    REQUIRE( timing.size() == nPoke * 4 );

    for( const timingRecord &rec : timing )
    {
        REQUIRE( rec[timingCol::sciN] == nFrames );
        REQUIRE( rec[timingCol::sciMissed] == 0 );

        // About 50 camwfs frames per camsci1 frame
        double perFrame = rec[timingCol::wfsN] / nFrames;
        REQUIRE( perFrame > 40 );
        REQUIRE( perFrame < 60 );

        if( !std::isnan( rec[timingCol::dmProj] ) )
        {
            REQUIRE( fabs( rec[timingCol::dmProj] - 1 ) < 1e-4 );
        }
    }

    // The second '-' of '+--+' needs no DM write
    REQUIRE( std::isnan( timing[2][timingCol::dmProj] ) );
    REQUIRE_FALSE( std::isnan( timing[3][timingCol::dmProj] ) );

    // The DM channel is restored to its initial (zero) content
    mx::improc::milkImage<float> dm( cfg.dmChannel );
    REQUIRE( dm().abs().maxCoeff() == 0 );
    dm.close();

    REQUIRE( std::filesystem::exists( dir / ( "test_" + cfg.sciName + "_resp.fits" ) ) );
    REQUIRE( std::filesystem::exists( dir / ( "test_" + cfg.wfsName + "_resp.fits" ) ) );
    REQUIRE( std::filesystem::exists( dir / "test_timing.fits" ) );

    mx::improc::eigenCube<float> sciFile;
    REQUIRE( ff.read( sciFile, ( dir / ( "test_" + cfg.sciName + "_resp.fits" ) ).string() ) == mx::error_t::noerror );
    REQUIRE( sciFile.planes() == nPoke );
    size_t nSci = app.sciResp().rows() * app.sciResp().cols() * nPoke;
    REQUIRE( ( Eigen::Map<const Eigen::ArrayXf>( sciFile.data(), nSci ) -
               Eigen::Map<const Eigen::ArrayXf>( app.sciResp().data(), nSci ) )
                 .abs()
                 .maxCoeff() == 0 );

    sim.stop();
    std::filesystem::remove_all( dir );
}

/// A stop request (as from SIGINT) mid-run restores the DM channel, writes the completed planes,
/// and returns 1.
TEST_CASE( "xlinqCal stop request restores DM and saves completed planes", "[xlinqCal][integration]" )
{
    using namespace xlinqCal;

    xlinqSimConfig cfg;
    cfg.prefix       = xlinqCalTest::testPrefix() + "stop_";
    cfg.dmChannel    = cfg.prefix + "dm01disp04";
    cfg.dmComb       = cfg.prefix + "dm01disp";
    cfg.wfsName      = cfg.prefix + "camwfs";
    cfg.sciName      = cfg.prefix + "camsci1";
    cfg.wfsFps       = 1000;
    cfg.sciFps       = 20;
    cfg.sciPostDelay = 0.015;

    std::filesystem::path dir = std::filesystem::temp_directory_path() / ( cfg.prefix + "out" );
    std::filesystem::create_directories( dir );

    std::string               pokeFile = ( dir / "pokes.fits" ).string();
    mx::fits::fitsFile<float> ff;
    REQUIRE( ff.write( pokeFile, xlinqSim::randomPokes( cfg.dmRows, cfg.dmCols, 20, 3 ) ) == mx::error_t::noerror );

    xlinqSim sim( cfg );
    sim.start();

    // A nonzero initial channel, which must be restored
    mx::improc::milkImage<float>  dm( cfg.dmChannel );
    mx::improc::eigenImage<float> initial( cfg.dmRows, cfg.dmCols );
    initial.setConstant( 0.01 );
    dm = initial;

    std::vector<std::string> args = { "xlinqCal",
                                      "--dm.channel=" + cfg.dmChannel,
                                      "--dm.comb=" + cfg.dmComb,
                                      "--sci.shmim=" + cfg.sciName,
                                      "--wfs.shmim=" + cfg.wfsName,
                                      "--poke.file=" + pokeFile,
                                      "--poke.amp=0.1",
                                      "--poke.first=5",
                                      "--sci.nFrames=2",
                                      "--preflight.time=0.3",
                                      "--out.dir=" + dir.string(),
                                      "--out.prefix=stop",
                                      "--yes" };

    std::vector<char *> argv;
    for( std::string &a : args )
    {
        argv.push_back( a.data() );
    }

    xlinqCal::xlinqCal app;
    std::thread        stopper(
        []()
        {
            std::this_thread::sleep_for( std::chrono::milliseconds( 2000 ) );
            xlinqCal::xlinqCal::requestStop();
        } );

    int rv = app.main( argv.size(), argv.data() );
    stopper.join();

    REQUIRE( rv == 1 );
    REQUIRE( app.nDone() > 0 );
    REQUIRE( app.nDone() < 15 );

    REQUIRE( ( dm() - initial ).abs().maxCoeff() == 0 );

    mx::improc::eigenCube<float> wfsFile;
    mx::fits::fitsHeader<>       head;
    REQUIRE( ff.read( wfsFile, head, ( dir / ( "stop_" + cfg.wfsName + "_resp.fits" ) ).string() ) ==
             mx::error_t::noerror );
    REQUIRE( wfsFile.planes() == app.nDone() );
    REQUIRE( head["POKE0"].value<int>() == 5 );
    REQUIRE( head["NPOKE"].value<int>() == app.nDone() );
    REQUIRE( head["COMPLETE"].value<int>() == 0 );

    dm.close();
    sim.stop();
    std::filesystem::remove_all( dir );
}
