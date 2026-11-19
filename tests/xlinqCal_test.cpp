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

/// Reference interpolation brackets a time when possible and falls back to the nearest reference.
TEST_CASE( "refWeights interpolates between bracketing references", "[xlinqCal]" )
{
    using xlinqCal::xlinqCal;

    size_t i0, i1;
    double w1;

    std::vector<double> t = { 10, 20, 40 };

    REQUIRE( xlinqCal::refWeights( t, 15, i0, i1, w1 ) );
    REQUIRE( i0 == 0 );
    REQUIRE( i1 == 1 );
    REQUIRE( w1 == Approx( 0.5 ) );

    REQUIRE( xlinqCal::refWeights( t, 35, i0, i1, w1 ) );
    REQUIRE( i0 == 1 );
    REQUIRE( i1 == 2 );
    REQUIRE( w1 == Approx( 0.75 ) );

    REQUIRE_FALSE( xlinqCal::refWeights( t, 5, i0, i1, w1 ) );
    REQUIRE( i0 == 0 );
    REQUIRE( i1 == 0 );
    REQUIRE( w1 == 0 );

    REQUIRE_FALSE( xlinqCal::refWeights( t, 50, i0, i1, w1 ) );
    REQUIRE( i0 == 2 );
    REQUIRE( i1 == 2 );

    REQUIRE_FALSE( xlinqCal::refWeights( {}, 50, i0, i1, w1 ) );
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

/// End to end against the simulator: recovered odd responses match amp * R * shape for both cameras,
/// the camsci1 even part matches amp^2 |H shape|^2 and the (linear) camwfs even part is zero despite
/// drift, the timing table and references are consistent, and the DM channel is restored.
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
    cfg.sciQuadGain  = 100;
    cfg.modevalName  = cfg.prefix + "modevalWFS";
    cfg.nModes       = 32;

    const int   nPoke   = 3;
    const float amp     = 0.1;
    const int   nFrames = 4;

    std::filesystem::path dir = std::filesystem::temp_directory_path() / ( cfg.prefix + "out" );
    std::filesystem::create_directories( dir );

    mx::improc::eigenCube<float> pokes    = xlinqSim::randomPokes( cfg.dmRows, cfg.dmCols, nPoke, 7 );
    std::string                  pokeFile = ( dir / "pokes.fits" ).string();
    mx::fits::fitsFile<float>    ff;
    REQUIRE( ff.write( pokeFile, pokes ) == mx::error_t::noerror );

    // Snapshot streams, standing in for aol1_wfsref and aol1_wfsmask
    std::string snapRef  = cfg.prefix + "wfsref";
    std::string snapMask = cfg.prefix + "wfsmask";

    simStream wfsref, wfsmask;
    wfsref.create( snapRef, _DATATYPE_FLOAT, 16, 16 );
    wfsmask.create( snapMask, _DATATYPE_FLOAT, 16, 16 );

    for( int i = 0; i < 256; ++i )
    {
        wfsref.data<float>()[i]  = 0.001f * i;
        wfsmask.data<float>()[i] = ( i % 3 == 0 ) ? 1.0f : 0.0f;
    }

    xlinqSim sim( cfg );
    sim.start();

    std::vector<std::string> args = { "xlinqCal",
                                      "--dm.channel=" + cfg.dmChannel,
                                      "--dm.comb=" + cfg.dmComb,
                                      "--sci.shmim=" + cfg.sciName,
                                      "--wfs.shmim=" + cfg.wfsName + "," + cfg.modevalName,
                                      "--poke.file=" + pokeFile,
                                      "--poke.amp=" + std::to_string( amp ),
                                      "--sci.nFrames=" + std::to_string( nFrames ),
                                      "--preflight.time=0.5",
                                      "--out.dir=" + dir.string(),
                                      "--out.prefix=test",
                                      "--snap.shmim=" + snapRef + "," + snapMask,
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
    REQUIRE( app.nWfs() == 2 );

    for( int k = 0; k < nPoke; ++k )
    {
        Eigen::Map<const Eigen::VectorXf> shape( pokes.image( k ).data(), cfg.dmRows * cfg.dmCols );

        Eigen::VectorXf sciExp = amp * ( sim.sciR().matrix() * shape );
        Eigen::VectorXf wfsExp = amp * ( sim.wfsR().matrix() * shape );

        Eigen::VectorXf modExp = amp * ( sim.modevalR().matrix() * shape );

        Eigen::Map<const Eigen::VectorXf> sciGot( app.sciProducts().resp.image( k ).data(), sciExp.size() );
        Eigen::Map<const Eigen::VectorXf> wfsGot( app.wfsProducts( 0 ).resp.image( k ).data(), wfsExp.size() );
        Eigen::Map<const Eigen::VectorXf> modGot( app.wfsProducts( 1 ).resp.image( k ).data(), modExp.size() );

        double sciErr = ( sciGot - sciExp ).norm() / sciExp.norm();
        double wfsErr = ( wfsGot - wfsExp ).norm() / wfsExp.norm();
        double modErr = ( modGot - modExp ).norm() / modExp.norm();

        INFO( "poke " << k << " sci err " << sciErr << " wfs err " << wfsErr << " modeval err " << modErr );
        REQUIRE( sciErr < 0.05 );
        REQUIRE( wfsErr < 0.05 );
        REQUIRE( modErr < 0.05 );
    }

    // Even parts, from the output files
    mx::improc::eigenCube<float> sciEven, wfsEven, sciRef, modEven;
    REQUIRE( ff.read( modEven, ( dir / ( "test_" + cfg.modevalName + "_even.fits" ) ).string() ) ==
             mx::error_t::noerror );
    REQUIRE( modEven.planes() == nPoke );
    REQUIRE( ff.read( sciEven, ( dir / ( "test_" + cfg.sciName + "_even.fits" ) ).string() ) == mx::error_t::noerror );
    REQUIRE( ff.read( wfsEven, ( dir / ( "test_" + cfg.wfsName + "_even.fits" ) ).string() ) == mx::error_t::noerror );
    REQUIRE( ff.read( sciRef, ( dir / ( "test_" + cfg.sciName + "_ref.fits" ) ).string() ) == mx::error_t::noerror );
    REQUIRE( sciEven.planes() == nPoke );
    REQUIRE( wfsEven.planes() == nPoke );
    REQUIRE( app.nRef() == nPoke + 1 );
    REQUIRE( sciRef.planes() == nPoke + 1 );

    for( int k = 0; k < nPoke; ++k )
    {
        Eigen::Map<const Eigen::VectorXf> shape( pokes.image( k ).data(), cfg.dmRows * cfg.dmCols );

        Eigen::ArrayXf sciExp =
            amp * amp *
            ( ( sim.sciHr().matrix() * shape ).array().square() + ( sim.sciHi().matrix() * shape ).array().square() );
        Eigen::VectorXf wfsOdd = amp * ( sim.wfsR().matrix() * shape );

        Eigen::Map<const Eigen::ArrayXf> sciGot( sciEven.image( k ).data(), sciExp.size() );
        Eigen::Map<const Eigen::ArrayXf> wfsGot( wfsEven.image( k ).data(), wfsOdd.size() );

        double sciErr = ( sciGot - sciExp ).matrix().norm() / sciExp.matrix().norm();
        double wfsRel = wfsGot.matrix().norm() / wfsOdd.norm();

        Eigen::VectorXf                  modOdd = amp * ( sim.modevalR().matrix() * shape );
        Eigen::Map<const Eigen::ArrayXf> modGot( modEven.image( k ).data(), modOdd.size() );
        double                           modRel = modGot.matrix().norm() / modOdd.norm();

        INFO( "poke " << k << " sci even err " << sciErr << " wfs even/odd " << wfsRel << " modeval even/odd "
                      << modRel );
        REQUIRE( sciErr < 0.05 );
        REQUIRE( wfsRel < 0.02 );
        REQUIRE( modRel < 0.02 );
    }

    const std::vector<timingRecord> &timing = app.timing();
    REQUIRE( timing.size() == nPoke * 4 + nPoke + 1 );

    for( const timingRecord &rec : timing )
    {
        REQUIRE( rec.size() == timingCol::nCommon + 2 * wfsCol::nCols );
        REQUIRE( rec[timingCol::sciN] == nFrames );
        REQUIRE( rec[timingCol::sciMissed] == 0 );

        // About 50 frames per camsci1 frame, for both WFS streams
        for( size_t j = 0; j < 2; ++j )
        {
            double perFrame = rec[wfsColIndex( j, wfsCol::n )] / nFrames;
            REQUIRE( perFrame > 40 );
            REQUIRE( perFrame < 60 );
        }

        if( !std::isnan( rec[timingCol::dmProj] ) )
        {
            REQUIRE( fabs( rec[timingCol::dmProj] - 1 ) < 1e-4 );
        }
    }

    // Rows: ref, + - - +, ref, ... The first ref and the second '-' need no DM write.
    REQUIRE( timing[0][timingCol::sign] == 0 );
    REQUIRE( std::isnan( timing[0][timingCol::dmProj] ) );
    REQUIRE_FALSE( std::isnan( timing[1][timingCol::dmProj] ) );
    REQUIRE( std::isnan( timing[3][timingCol::dmProj] ) );
    REQUIRE_FALSE( std::isnan( timing[4][timingCol::dmProj] ) );
    REQUIRE( timing[5][timingCol::sign] == 0 );
    REQUIRE_FALSE( std::isnan( timing[5][timingCol::dmProj] ) );

    // The DM channel is restored to its initial (zero) content
    mx::improc::milkImage<float> dm( cfg.dmChannel );
    REQUIRE( dm().abs().maxCoeff() == 0 );
    dm.close();

    REQUIRE( std::filesystem::exists( dir / ( "test_" + cfg.sciName + "_resp.fits" ) ) );
    REQUIRE( std::filesystem::exists( dir / ( "test_" + cfg.wfsName + "_resp.fits" ) ) );
    REQUIRE( std::filesystem::exists( dir / "test_timing.fits" ) );

    // Snapshots match the streams, and are unchanged so no _end files
    mx::improc::eigenImage<float> snapIm;
    REQUIRE( ff.read( snapIm, ( dir / ( "test_snap_" + snapRef + ".fits" ) ).string() ) == mx::error_t::noerror );
    REQUIRE( snapIm.rows() == 16 );
    REQUIRE( snapIm( 5, 1 ) == Approx( 0.001f * ( 5 + 16 ) ) );
    REQUIRE( ff.read( snapIm, ( dir / ( "test_snap_" + snapMask + ".fits" ) ).string() ) == mx::error_t::noerror );
    REQUIRE( snapIm.sum() == 86 );
    REQUIRE_FALSE( std::filesystem::exists( dir / ( "test_snap_" + snapRef + "_end.fits" ) ) );
    REQUIRE_FALSE( std::filesystem::exists( dir / ( "test_snap_" + snapMask + "_end.fits" ) ) );

    mx::improc::eigenCube<float> sciFile;
    REQUIRE( ff.read( sciFile, ( dir / ( "test_" + cfg.sciName + "_resp.fits" ) ).string() ) == mx::error_t::noerror );
    REQUIRE( sciFile.planes() == nPoke );
    const mx::improc::eigenCube<float> &sciResp = app.sciProducts().resp;
    size_t                              nSci    = sciResp.rows() * sciResp.cols() * nPoke;
    REQUIRE( ( Eigen::Map<const Eigen::ArrayXf>( sciFile.data(), nSci ) -
               Eigen::Map<const Eigen::ArrayXf>( sciResp.data(), nSci ) )
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

    // A snapshot stream that changes during the run
    std::string snapRef = cfg.prefix + "wfsref";
    simStream   wfsref;
    wfsref.create( snapRef, _DATATYPE_FLOAT, 8, 8 );

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
                                      "--snap.shmim=" + snapRef,
                                      "--yes" };

    std::vector<char *> argv;
    for( std::string &a : args )
    {
        argv.push_back( a.data() );
    }

    xlinqCal::xlinqCal app;
    std::thread        stopper(
        [&wfsref]()
        {
            std::this_thread::sleep_for( std::chrono::milliseconds( 1000 ) );
            wfsref.data<float>()[3] = 7;
            wfsref.post();
            std::this_thread::sleep_for( std::chrono::milliseconds( 1000 ) );
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

    mx::improc::eigenCube<float> evenFile;
    REQUIRE( ff.read( evenFile, ( dir / ( "stop_" + cfg.wfsName + "_even.fits" ) ).string() ) == mx::error_t::noerror );
    REQUIRE( evenFile.planes() == app.nDone() );
    REQUIRE( app.nRef() >= app.nDone() );

    // The changed snapshot stream is saved at the start (zeros) and at the end
    mx::improc::eigenImage<float> snapIm;
    REQUIRE( ff.read( snapIm, ( dir / ( "stop_snap_" + snapRef + ".fits" ) ).string() ) == mx::error_t::noerror );
    REQUIRE( snapIm.abs().maxCoeff() == 0 );
    REQUIRE( ff.read( snapIm, ( dir / ( "stop_snap_" + snapRef + "_end.fits" ) ).string() ) == mx::error_t::noerror );
    REQUIRE( snapIm( 3, 0 ) == 7 );

    dm.close();
    sim.stop();
    std::filesystem::remove_all( dir );
}
