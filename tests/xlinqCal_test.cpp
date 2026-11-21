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

/// Doubled names, the signature of a repeated config key joined by mxlib, are detected.
TEST_CASE( "looksDoubled detects repeated-key stream names", "[xlinqCal]" )
{
    using xlinqCal::xlinqCal;

    REQUIRE( xlinqCal::looksDoubled( "aol1_imWFS2pubaol1_imWFS2pub" ) );
    REQUIRE_FALSE( xlinqCal::looksDoubled( "aol1_imWFS2pub" ) );
    REQUIRE_FALSE( xlinqCal::looksDoubled( "camsci1" ) );
    REQUIRE_FALSE( xlinqCal::looksDoubled( "" ) );
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
    mx::fits::fitsHeader<>       pokeHead;
    pokeHead.append( "TESTCARD", 42, "should be copied to _pokes.fits" );
    REQUIRE( ff.write( pokeFile, pokes, pokeHead ) == mx::error_t::noerror );

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
                                      "--out.timestamp=false",
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

        // Acquisition times: the averaged camsci1 exposures end inside the gate, one frame period apart
        double t0 = rec[timingCol::tStart];
        double t1 = rec[timingCol::tEnd];
        double sa = rec[timingCol::sciAtimeFirst];
        double sb = rec[timingCol::sciAtimeLast];
        INFO( "gate " << t0 << "-" << t1 << " sci atime " << sa << "-" << sb );
        REQUIRE( sa > t0 - 0.05 );
        REQUIRE( sb <= t1 );
        REQUIRE( fabs( ( sb - sa ) - ( nFrames - 1 ) / cfg.sciFps ) < 0.2 / cfg.sciFps );

        // WFS frames fall inside the gate; modeval carries the camwfs atime
        for( size_t j = 0; j < 2; ++j )
        {
            REQUIRE( rec[wfsColIndex( j, wfsCol::atimeFirst )] > t0 - 0.01 );
            REQUIRE( rec[wfsColIndex( j, wfsCol::atimeLast )] <= t1 + 0.01 );
            REQUIRE( rec[wfsColIndex( j, wfsCol::atimeLast )] > rec[wfsColIndex( j, wfsCol::atimeFirst )] );
        }
        REQUIRE( fabs( rec[wfsColIndex( 1, wfsCol::atimeFirst )] - rec[wfsColIndex( 0, wfsCol::atimeFirst )] ) <
                 2.0 / cfg.wfsFps );

        REQUIRE( rec[timingCol::sciMax] > cfg.sciLevel );
    }

    REQUIRE( app.nNonlin() == 0 );

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

    // Applied pokes are saved with the poke file's header
    mx::improc::eigenCube<float> applied;
    mx::fits::fitsHeader<>       appliedHead;
    REQUIRE( ff.read( applied, appliedHead, ( dir / "test_pokes.fits" ).string() ) == mx::error_t::noerror );
    REQUIRE( applied.planes() == nPoke );
    REQUIRE( ( applied.image( 2 ) - amp * pokes.image( 2 ) ).abs().maxCoeff() < 1e-6 );
    REQUIRE( appliedHead["TESTCARD"].value<int>() == 42 );
    REQUIRE( appliedHead["POKEAMP"].value<float>() == Approx( amp ) );

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

    // A second run with the same prefix is refused before anything is written or poked
    auto respTime = std::filesystem::last_write_time( dir / ( "test_" + cfg.sciName + "_resp.fits" ) );
    auto logSize  = std::filesystem::file_size( dir / "test.log" );
    {
        xlinqCal::xlinqCal again;
        REQUIRE( again.main( argv.size(), argv.data() ) == -1 );
        REQUIRE( again.nDone() == 0 );
    }
    REQUIRE( std::filesystem::last_write_time( dir / ( "test_" + cfg.sciName + "_resp.fits" ) ) == respTime );
    REQUIRE( std::filesystem::file_size( dir / "test.log" ) == logSize );

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
                                      "--sci.linearMax=1",
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

    // Every completed acquisition is above the (deliberately low) linearity limit
    size_t nAcq = 0;
    for( const timingRecord &rec : app.timing() )
    {
        nAcq += !std::isnan( rec[timingCol::sciMax] );
    }
    REQUIRE( app.nNonlin() == static_cast<int>( nAcq ) );

    REQUIRE( ( dm() - initial ).abs().maxCoeff() == 0 );

    // The default prefix is out.prefix_<UTC timestamp>
    std::string pre = std::filesystem::path( app.outputPrefix() ).filename().string();
    REQUIRE( pre.size() == std::string( "stop_YYYYMMDDTHHMMSS" ).size() );
    REQUIRE( pre.compare( 0, 5, "stop_" ) == 0 );
    REQUIRE( pre[13] == 'T' );
    pre += "_";

    mx::improc::eigenCube<float> wfsFile;
    mx::fits::fitsHeader<>       head;
    REQUIRE( ff.read( wfsFile, head, ( dir / ( pre + cfg.wfsName + "_resp.fits" ) ).string() ) ==
             mx::error_t::noerror );
    REQUIRE( wfsFile.planes() == app.nDone() );
    REQUIRE( head["POKE0"].value<int>() == 5 );
    REQUIRE( head["NPOKE"].value<int>() == app.nDone() );
    REQUIRE( head["COMPLETE"].value<int>() == 0 );

    mx::improc::eigenCube<float> evenFile;
    REQUIRE( ff.read( evenFile, ( dir / ( pre + cfg.wfsName + "_even.fits" ) ).string() ) == mx::error_t::noerror );
    REQUIRE( evenFile.planes() == app.nDone() );
    REQUIRE( app.nRef() >= app.nDone() );

    // The changed snapshot stream is saved at the start (zeros) and at the end
    mx::improc::eigenImage<float> snapIm;
    REQUIRE( ff.read( snapIm, ( dir / ( pre + "snap_" + snapRef + ".fits" ) ).string() ) == mx::error_t::noerror );
    REQUIRE( snapIm.abs().maxCoeff() == 0 );
    REQUIRE( ff.read( snapIm, ( dir / ( pre + "snap_" + snapRef + "_end.fits" ) ).string() ) == mx::error_t::noerror );
    REQUIRE( snapIm( 3, 0 ) == 7 );

    dm.close();
    sim.stop();
    std::filesystem::remove_all( dir );
}

/// runCommand captures output and exit status, runs without a shell, and kills at the timeout.
TEST_CASE( "runCommand captures output, status, and timeouts", "[indiClient]" )
{
    using namespace xlinqCal;

    cmdResult r = runCommand( { "echo", "a b;c" }, 5 );
    REQUIRE( r.status == 0 );
    REQUIRE_FALSE( r.timedOut );
    REQUIRE( r.out == "a b;c\n" ); // no shell: ';' is literal

    r = runCommand( { "sh", "-c", "echo oops >&2; exit 3" }, 5 );
    REQUIRE( r.status == 3 );
    REQUIRE( r.err == "oops\n" );

    auto t0   = std::chrono::steady_clock::now();
    r         = runCommand( { "sleep", "5" }, 0.3 );
    double dt = std::chrono::duration<double>( std::chrono::steady_clock::now() - t0 ).count();
    REQUIRE( r.timedOut );
    REQUIRE( r.status == -1 );
    REQUIRE( dt < 2 );

    r = runCommand( { "/nonexistent/xlsim_no_such_command" }, 5 );
    REQUIRE( r.status == 127 );
}

/// INDI names are exact device.property[.element] with no wildcards or shell characters.
TEST_CASE( "indiClient validates names", "[indiClient]" )
{
    using xlinqCal::indiClient;

    REQUIRE( indiClient::validName( "camsci1.exptime", 1 ) );
    REQUIRE( indiClient::validName( "camsci1.shutter_status.status", 2 ) );
    REQUIRE( indiClient::validName( "fw-sci1.filter", 1 ) );
    REQUIRE_FALSE( indiClient::validName( "camsci1", 1 ) );
    REQUIRE_FALSE( indiClient::validName( "camsci1.exptime", 2 ) );
    REQUIRE_FALSE( indiClient::validName( "camsci1.*", 1 ) );
    REQUIRE_FALSE( indiClient::validName( "*.exptime", 1 ) );
    REQUIRE_FALSE( indiClient::validName( "camsci1..exptime", 2 ) );
    REQUIRE_FALSE( indiClient::validName( "camsci1.exptime.", 2 ) );
    REQUIRE_FALSE( indiClient::validName( "cam sci1.exptime", 1 ) );
    REQUIRE_FALSE( indiClient::validName( "camsci1.exptime;rm", 1 ) );
}

/// \cond
namespace xlinqCalTest
{

/// Fake getINDI/setINDI scripts over a state directory of `device.property.element` files.
/** setINDI requires a type flag and applies the new value after 0.3 s, like an asynchronous device.
 * Calls are logged to getlog and setlog. Nothing here talks to an indiserver.
 */
struct fakeIndi
{
    std::filesystem::path dir;   ///< The base directory.
    std::filesystem::path state; ///< The state directory.
    std::string           getCmd;
    std::string           setCmd;

    explicit fakeIndi( const std::filesystem::path &base ) : dir( base ), state( base / "state" )
    {
        std::filesystem::create_directories( state );
        getCmd = ( dir / "getINDI" ).string();
        setCmd = ( dir / "setINDI" ).string();

        std::ofstream g( getCmd );
        g << "#!/bin/bash\n"
             "S='"
          << state.string()
          << "'\n"
             "spec=\"${@: -1}\"\n"
             "echo \"$spec\" >> '"
          << ( dir / "getlog" ).string()
          << "'\n"
             "found=0\n"
             "for f in \"$S/$spec\".*; do [ -e \"$f\" ] || continue; echo \"$(basename \"$f\")=$(cat \"$f\")\"; "
             "found=1; done\n"
             "[ $found = 1 ] || { echo \"No $spec from localhost:7624\" >&2; exit 1; }\n";
        g.close();

        std::ofstream s( setCmd );
        s << "#!/bin/bash\n"
             "S='"
          << state.string()
          << "'\n"
             "[ \"$1\" = -s ] || [ \"$1\" = -n ] || { echo 'untyped set' >&2; exit 1; }\n"
             "arg=\"$2\"; dpe=\"${arg%%=*}\"; val=\"${arg#*=}\"\n"
             "echo \"$1 $arg\" >> '"
          << ( dir / "setlog" ).string()
          << "'\n"
             "prop=\"${dpe%.*}\"\n"
             // A one-of-many switch (marked by a 1ofmany.<device.property> file) turns its other elements Off
             "( sleep 0.3; if [ \"$val\" = On ] && [ -e \"$S/1ofmany.$prop\" ]; then for f in \"$S/$prop\".*; do "
             "echo Off > \"$S/../tmp.$$\" && mv \"$S/../tmp.$$\" \"$f\"; done; fi; "
             "echo \"$val\" > \"$S/../tmp.$$\" && mv \"$S/../tmp.$$\" \"$S/$dpe\" ) >/dev/null 2>&1 &\n";
        s.close();

        std::filesystem::permissions( getCmd, std::filesystem::perms::owner_all );
        std::filesystem::permissions( setCmd, std::filesystem::perms::owner_all );
    }

    void set( const std::string &dpe, const std::string &val )
    {
        std::ofstream( state / dpe ) << val << "\n";
    }

    /// Mark `device.property` as a one-of-many switch.
    void oneOfMany( const std::string &dp )
    {
        std::ofstream( state / ( "1ofmany." + dp ) ) << "\n";
    }

    std::string get( const std::string &dpe ) const
    {
        std::ifstream f( state / dpe );
        std::string   v;
        f >> v;
        return v;
    }

    std::vector<std::string> lines( const std::string &log ) const
    {
        std::vector<std::string> out;
        std::ifstream            f( dir / log );
        std::string              l;
        while( std::getline( f, l ) )
        {
            out.push_back( l );
        }
        return out;
    }
};

} // namespace xlinqCalTest
/// \endcond

/// With INDI enabled, xlinqCal records camera state at start and end, takes a dark with the shutter
/// shut after the run, and leaves the shutter open. All INDI calls are exact and type-flagged.
TEST_CASE( "xlinqCal records INDI state and takes a dark", "[xlinqCal][integration][indi]" )
{
    using namespace xlinqCal;

    std::filesystem::path dir = std::filesystem::temp_directory_path() / ( xlinqCalTest::testPrefix() + "indi_out" );
    std::filesystem::create_directories( dir );
    xlinqCalTest::fakeIndi indi( dir / "indi" );

    const std::string dev = "xlsimcam";
    indi.set( dev + ".fsm.state", "READY" );
    indi.set( dev + ".exptime.current", "0.05" );
    indi.set( dev + ".exptime.target", "0.05" );
    indi.set( dev + ".emgain.current", "1" );
    indi.set( dev + ".fps.current", "20" );
    indi.set( dev + ".shutter_status.status", "READY" );
    indi.set( dev + ".shutter.toggle", "Off" );

    xlinqSimConfig cfg;
    cfg.prefix       = xlinqCalTest::testPrefix() + "indi_";
    cfg.dmChannel    = cfg.prefix + "dm01disp04";
    cfg.dmComb       = cfg.prefix + "dm01disp";
    cfg.wfsName      = cfg.prefix + "camwfs";
    cfg.sciName      = cfg.prefix + "camsci1";
    cfg.wfsFps       = 1000;
    cfg.sciFps       = 20;
    cfg.sciPostDelay = 0.015;
    cfg.shutterFile  = ( indi.state / ( dev + ".shutter.toggle" ) ).string();

    mx::improc::eigenCube<float> pokes    = xlinqSim::randomPokes( cfg.dmRows, cfg.dmCols, 2, 3 );
    std::string                  pokeFile = ( dir / "pokes.fits" ).string();
    mx::fits::fitsFile<float>    ff;
    REQUIRE( ff.write( pokeFile, pokes ) == mx::error_t::noerror );

    xlinqSim sim( cfg );
    sim.start();

    auto runApp = [&]( const std::string &prefix, std::vector<std::string> extra )
    {
        std::vector<std::string> args = { "xlinqCal",
                                          "--dm.channel=" + cfg.dmChannel,
                                          "--dm.comb=" + cfg.dmComb,
                                          "--sci.shmim=" + cfg.sciName,
                                          "--wfs.shmim=" + cfg.wfsName,
                                          "--poke.file=" + pokeFile,
                                          "--poke.amp=0.1",
                                          "--sci.nFrames=2",
                                          "--preflight.time=0.3",
                                          "--out.dir=" + dir.string(),
                                          "--out.prefix=" + prefix,
                                          "--out.timestamp=false",
                                          "--indi.enable=true",
                                          "--indi.get=" + indi.getCmd,
                                          "--indi.set=" + indi.setCmd,
                                          "--indi.poll=0.2",
                                          "--sci.device=" + dev,
                                          "--yes" };
        args.insert( args.end(), extra.begin(), extra.end() );

        std::vector<char *> argv;
        for( std::string &a : args )
        {
            argv.push_back( a.data() );
        }

        xlinqCal::xlinqCal app;
        int                rv = app.main( argv.size(), argv.data() );
        return std::make_pair( rv, app.nDone() );
    };

    SECTION( "complete run with a dark" )
    {
        auto [rv, nDone] = runApp( "indi", { "--dark.n=3", "--indi.props=" + dev + ".missing" } );
        REQUIRE( rv == 0 );
        REQUIRE( nDone == 2 );

        // State files: camera properties at start and end, with the unreadable one noted
        std::ifstream     fin( dir / "indi_indi.txt" );
        std::stringstream ss;
        ss << fin.rdbuf();
        REQUIRE( ss.str().find( dev + ".exptime.current=0.05" ) != std::string::npos );
        REQUIRE( ss.str().find( dev + ".shutter.toggle=Off" ) != std::string::npos );
        REQUIRE( ss.str().find( "# " + dev + ".missing:" ) != std::string::npos );
        REQUIRE( std::filesystem::exists( dir / "indi_indi_end.txt" ) );

        // The dark is the shut level, not the open level
        mx::improc::eigenImage<float> dark;
        mx::fits::fitsHeader<>        dhead;
        REQUIRE( ff.read( dark, dhead, ( dir / ( "indi_" + cfg.sciName + "_dark.fits" ) ).string() ) ==
                 mx::error_t::noerror );
        REQUIRE( dark.mean() == Approx( cfg.sciDark ).margin( 2 ) );
        REQUIRE( dhead["NDARK"].value<int>() == 3 );

        mx::improc::eigenCube<float> resp;
        mx::fits::fitsHeader<>       head;
        REQUIRE( ff.read( resp, head, ( dir / ( "indi_" + cfg.sciName + "_resp.fits" ) ).string() ) ==
                 mx::error_t::noerror );
        REQUIRE( head["SCIEXPT"].value<double>() == Approx( 0.05 ) );
        REQUIRE( head["NDARK"].value<int>() == 3 );

        // The shutter is open again, and was commanded shut then open with typed sets
        REQUIRE( indi.get( dev + ".shutter.toggle" ) == "Off" );
        std::vector<std::string> sets = indi.lines( "setlog" );
        REQUIRE( sets.size() == 2 );
        REQUIRE( sets[0] == "-s " + dev + ".shutter.toggle=On" );
        REQUIRE( sets[1] == "-s " + dev + ".shutter.toggle=Off" );

        // Every get was one exact device.property
        for( const std::string &g : indi.lines( "getlog" ) )
        {
            INFO( g );
            REQUIRE( indiClient::validName( g, 1 ) );
        }
    }

    SECTION( "refuses to start with the shutter shut" )
    {
        indi.set( dev + ".shutter.toggle", "On" );
        auto [rv, nDone] = runApp( "shut", { "--dark.n=3" } );
        REQUIRE( rv == -1 );
        REQUIRE( nDone == 0 );
        REQUIRE( indi.lines( "setlog" ).empty() );
    }

    SECTION( "refuses a dark when the shutter is not ready" )
    {
        indi.set( dev + ".shutter_status.status", "POWEROFF" );
        auto [rv, nDone] = runApp( "notready", { "--dark.n=3" } );
        REQUIRE( rv == -1 );
        REQUIRE( nDone == 0 );
        REQUIRE( indi.lines( "setlog" ).empty() );
    }

    SECTION( "a dark needs indi.enable" )
    {
        auto [rv, nDone] = runApp( "noindi", { "--dark.n=3", "--indi.enable=false" } );
        REQUIRE( rv == -1 );
        REQUIRE( nDone == 0 );
        REQUIRE( indi.lines( "getlog" ).empty() );
    }

    sim.stop();
    std::filesystem::remove_all( dir );
}

/// With a saturated core, xlinqCal takes unsaturated PSFs with the ND wheel at the start and end, with +/- calibration
/// speckles at both filters whose ratio is the transmission, puts the wheel back each time, and excludes the core
/// from the linearity check. Already at the PSF filter, it takes the block in place. It refuses to saturate with
/// EM gain > 1.
TEST_CASE( "xlinqCal takes unsaturated PSFs with the ND wheel", "[xlinqCal][integration][indi]" )
{
    using namespace xlinqCal;

    std::filesystem::path dir = std::filesystem::temp_directory_path() / ( xlinqCalTest::testPrefix() + "psf_out" );
    std::filesystem::create_directories( dir );
    xlinqCalTest::fakeIndi indi( dir / "indi" );

    const std::string dev   = "xlsimcam";
    const std::string wheel = "xlsimfw";
    indi.set( dev + ".fsm.state", "READY" );
    indi.set( dev + ".exptime.current", "0.05" );
    indi.set( dev + ".emgain.current", "1" );
    indi.set( dev + ".shutter_status.status", "READY" );
    indi.set( dev + ".shutter.toggle", "Off" );
    indi.set( wheel + ".fsm.state", "READY" );
    indi.set( wheel + ".filterName.open", "Off" );
    indi.set( wheel + ".filterName.ND1_0", "On" );
    indi.set( wheel + ".filterName.ND3_0", "Off" );
    indi.oneOfMany( wheel + ".filterName" );

    xlinqSimConfig cfg;
    cfg.prefix       = xlinqCalTest::testPrefix() + "psf_";
    cfg.dmChannel    = cfg.prefix + "dm01disp04";
    cfg.dmComb       = cfg.prefix + "dm01disp";
    cfg.wfsName      = cfg.prefix + "camwfs";
    cfg.sciName      = cfg.prefix + "camsci1";
    cfg.wfsFps       = 1000;
    cfg.sciFps       = 20;
    cfg.sciPostDelay = 0.015;
    cfg.sciCoreAmp   = 60000;
    cfg.sciCoreSigma = 1.5;
    cfg.sciSat       = 50000;
    cfg.attenFile    = ( indi.state / ( wheel + ".filterName.ND3_0" ) ).string();
    cfg.attenFactor  = 0.01;
    cfg.sciQuadGain  = 200; // the calibration speckle intensity

    mx::improc::eigenCube<float> pokes    = xlinqSim::randomPokes( cfg.dmRows, cfg.dmCols, 2, 3 );
    std::string                  pokeFile = ( dir / "pokes.fits" ).string();
    mx::fits::fitsFile<float>    ff;
    REQUIRE( ff.write( pokeFile, pokes ) == mx::error_t::noerror );

    xlinqSim sim( cfg );
    sim.start();

    struct result
    {
        int rv;
        int nDone;
        int nNonlin;
    };

    auto runApp = [&]( const std::string &prefix, std::vector<std::string> extra )
    {
        std::vector<std::string> args = { "xlinqCal",
                                          "--dm.channel=" + cfg.dmChannel,
                                          "--dm.comb=" + cfg.dmComb,
                                          "--sci.shmim=" + cfg.sciName,
                                          "--wfs.shmim=" + cfg.wfsName,
                                          "--poke.file=" + pokeFile,
                                          "--poke.amp=0.1",
                                          "--sci.nFrames=2",
                                          "--preflight.time=0.3",
                                          "--out.dir=" + dir.string(),
                                          "--out.prefix=" + prefix,
                                          "--out.timestamp=false",
                                          "--indi.enable=true",
                                          "--indi.get=" + indi.getCmd,
                                          "--indi.set=" + indi.setCmd,
                                          "--indi.poll=0.2",
                                          "--sci.device=" + dev,
                                          "--sci.linearMax=37000",
                                          "--yes" };
        args.insert( args.end(), extra.begin(), extra.end() );

        std::vector<char *> argv;
        for( std::string &a : args )
        {
            argv.push_back( a.data() );
        }

        xlinqCal::xlinqCal app;
        int                rv = app.main( argv.size(), argv.data() );
        return result{ rv, app.nDone(), app.nNonlin() };
    };

    const std::vector<std::string> psfArgs = {
        "--psf.n=3", "--psf.wheel=" + wheel, "--psf.filter=ND3_0", "--psf.discard=1", "--sci.satRadius=4" };

    // Pixels outside the core mask, unsaturated in every plane: (+ + -)/2 - reference summed over them
    auto speckleSum = [&]( const mx::improc::eigenCube<float> &c, int p0, const mx::improc::eigenCube<float> &all )
    {
        double sum = 0;
        int    n   = 0;
        for( int cc = 0; cc < c.cols(); ++cc )
        {
            for( int rr = 0; rr < c.rows(); ++rr )
            {
                bool ok = std::hypot( rr - 5.5, cc - 5.5 ) > 4.5;
                for( int p = 0; p < all.planes(); ++p )
                {
                    ok = ok && all.image( p )( rr, cc ) < 37000;
                }
                if( ok )
                {
                    sum +=
                        0.5 * ( c.image( p0 + 1 )( rr, cc ) + c.image( p0 + 2 )( rr, cc ) ) - c.image( p0 )( rr, cc );
                    ++n;
                }
            }
        }
        REQUIRE( n > 50 );
        return sum;
    };

    SECTION( "saturated run with PSFs and calibration speckles at the start and end" )
    {
        std::vector<std::string> a = psfArgs;
        a.push_back( "--psf.speckAmp=0.5" );
        a.push_back( "--psf.speckSep=10" );
        result r = runApp( "psf", a );
        REQUIRE( r.rv == 0 );
        REQUIRE( r.nDone == 2 );
        REQUIRE( r.nNonlin == 0 ); // the saturated core is masked

        // Both PSFs are unsaturated, attenuated copies of the core, and the speckle ratio is the transmission
        double expectPeak =
            cfg.sciDark + cfg.attenFactor * ( cfg.sciLevel - cfg.sciDark + cfg.sciCoreAmp * exp( -0.25 / 2.25 ) );
        for( const std::string suf : { "_psf.fits", "_psf_end.fits" } )
        {
            mx::improc::eigenCube<float> psf;
            mx::fits::fitsHeader<>       head;
            INFO( suf );
            REQUIRE( ff.read( psf, head, ( dir / ( "psf_" + cfg.sciName + suf ) ).string() ) == mx::error_t::noerror );
            REQUIRE( psf.planes() == 6 );
            REQUIRE( psf.image( 0 ).maxCoeff() == Approx( expectPeak ).epsilon( 0.05 ) );
            REQUIRE( psf.image( 3 ).maxCoeff() == Approx( cfg.sciSat ).epsilon( 0.01 ) );
            REQUIRE( head["PSFFILT"].value<std::string>().substr( 0, 5 ) == "ND3_0" );
            REQUIRE( head["ORIGFILT"].value<std::string>().substr( 0, 5 ) == "ND1_0" );
            REQUIRE( head["PLANE4"].value<std::string>() == "ND1_0, DM reference + speckles" );
            REQUIRE( head["PSFMAX"].value<double>() < 37000 );
            REQUIRE( head["SPKAMP"].value<double>() == Approx( 0.5 ) );

            double sPsf = speckleSum( psf, 0, psf ), sCal = speckleSum( psf, 3, psf );
            REQUIRE( sCal > 0 );
            REQUIRE( sPsf / sCal == Approx( cfg.attenFactor ).epsilon( 0.03 ) );
        }

        // The calibration frames are saturated, and the header records the mask and the PSFs
        mx::improc::eigenCube<float> ref;
        mx::fits::fitsHeader<>       head;
        REQUIRE( ff.read( ref, head, ( dir / ( "psf_" + cfg.sciName + "_ref.fits" ) ).string() ) ==
                 mx::error_t::noerror );
        REQUIRE( ref.image( 0 ).maxCoeff() == Approx( cfg.sciSat ).epsilon( 0.01 ) );
        REQUIRE( head["NPSF"].value<int>() == 2 );
        REQUIRE( head["SATRAD"].value<double>() == 4 );
        REQUIRE( head["SATCX"].value<double>() == Approx( 5.5 ) );
        REQUIRE( head["SATCY"].value<double>() == Approx( 5.5 ) );

        // The wheel is back, after exactly two typed round trips
        REQUIRE( indi.get( wheel + ".filterName.ND1_0" ) == "On" );
        REQUIRE( indi.get( wheel + ".filterName.ND3_0" ) == "Off" );
        std::vector<std::string> sets = indi.lines( "setlog" );
        REQUIRE( sets.size() == 4 );
        for( size_t i = 0; i < sets.size(); ++i )
        {
            REQUIRE( sets[i] == "-s " + wheel + ".filterName." + ( i % 2 == 0 ? "ND3_0" : "ND1_0" ) + "=On" );
        }

        // The wheel state is in the INDI record, and every get was one exact device.property
        std::ifstream     fin( dir / "psf_indi.txt" );
        std::stringstream ss;
        ss << fin.rdbuf();
        REQUIRE( ss.str().find( wheel + ".filterName.ND1_0=On" ) != std::string::npos );
        for( const std::string &g : indi.lines( "getlog" ) )
        {
            INFO( g );
            REQUIRE( indiClient::validName( g, 1 ) );
        }
    }

    SECTION( "without the core mask the saturated core is counted" )
    {
        result r = runApp( "nomask", {} );
        REQUIRE( r.rv == 0 );
        REQUIRE( r.nNonlin > 0 );
        REQUIRE( indi.lines( "setlog" ).empty() );
    }

    SECTION( "takes the PSF block in place when the wheel is already at the PSF filter" )
    {
        indi.set( wheel + ".filterName.ND1_0", "Off" );
        indi.set( wheel + ".filterName.ND3_0", "On" );
        std::vector<std::string> a = psfArgs;
        a.push_back( "--psf.speckAmp=0.5" );
        result r = runApp( "atpsf", a );
        REQUIRE( r.rv == 0 );
        REQUIRE( r.nDone == 2 );
        REQUIRE( indi.lines( "setlog" ).empty() );

        mx::improc::eigenCube<float> psf;
        mx::fits::fitsHeader<>       head;
        REQUIRE( ff.read( psf, head, ( dir / ( "atpsf_" + cfg.sciName + "_psf_end.fits" ) ).string() ) ==
                 mx::error_t::noerror );
        REQUIRE( psf.planes() == 3 );
        REQUIRE( head["PSFMOVE"].value<int>() == 0 );
        REQUIRE( speckleSum( psf, 0, psf ) > 0 );
    }

    SECTION( "a PSF block without speckles is one plane" )
    {
        result r = runApp( "plain", psfArgs );
        REQUIRE( r.rv == 0 );
        mx::improc::eigenCube<float> psf;
        mx::fits::fitsHeader<>       head;
        REQUIRE( ff.read( psf, head, ( dir / ( "plain_" + cfg.sciName + "_psf.fits" ) ).string() ) ==
                 mx::error_t::noerror );
        REQUIRE( psf.planes() == 1 );
        REQUIRE( indi.lines( "setlog" ).size() == 4 );
    }

    SECTION( "low orders unsaturated at the PSF filter, the rest saturated" )
    {
        std::vector<std::string> a = psfArgs;
        a.push_back( "--psf.speckAmp=0.5" );
        a.push_back( "--psf.unsatLast=0" );
        result r = runApp( "split", a );
        REQUIRE( r.rv == 0 );
        REQUIRE( r.nDone == 2 );
        REQUIRE( r.nNonlin == 0 );

        // References: start and after poke 0 at ND3_0, then after the switch and after poke 1 at ND1_0
        mx::improc::eigenCube<float> ref, even;
        mx::fits::fitsHeader<>       head, ehead;
        REQUIRE( ff.read( ref, head, ( dir / ( "split_" + cfg.sciName + "_ref.fits" ) ).string() ) ==
                 mx::error_t::noerror );
        REQUIRE( ref.planes() == 4 );
        REQUIRE( head["UNSATLST"].value<int>() == 0 );
        REQUIRE( head["NREFUNS"].value<int>() == 2 );
        for( int p = 0; p < 4; ++p )
        {
            INFO( p );
            REQUIRE( ( ref.image( p ).maxCoeff() < 37000 ) == ( p < 2 ) );
        }

        // Each poke's even part uses references at its own filter: no saturated-minus-unsaturated core
        REQUIRE( ff.read( even, ehead, ( dir / ( "split_" + cfg.sciName + "_even.fits" ) ).string() ) ==
                 mx::error_t::noerror );
        for( int p = 0; p < 2; ++p )
        {
            INFO( p );
            REQUIRE( std::fabs( even.image( p )( 5, 5 ) ) < 200 );
            REQUIRE( std::fabs( even.image( p )( 6, 6 ) ) < 200 );
        }

        // The timing table flags the unsaturated acquisitions
        mx::improc::eigenImage<double> tab;
        mx::fits::fitsHeader<>         thead;
        mx::fits::fitsFile<double>     fd;
        REQUIRE( fd.read( tab, thead, ( dir / "split_timing.fits" ).string() ) == mx::error_t::noerror );
        REQUIRE( thead["TCOL" + std::to_string( timingCol::unsat )].value<std::string>().substr( 0, 5 ) == "UNSAT" );
        int nU = 0, nS = 0;
        for( int rr = 0; rr < tab.cols(); ++rr )
        {
            double pk = tab( timingCol::poke, rr );
            if( pk == pk ) // a poke, not a reference
            {
                REQUIRE( tab( timingCol::unsat, rr ) == ( pk == 0 ? 1 : 0 ) );
                ( pk == 0 ? nU : nS )++;
            }
        }
        REQUIRE( nU == 4 );
        REQUIRE( nS == 4 );

        // Wheel: start block, the segment, end block; back at ND1_0
        std::vector<std::string> sets = indi.lines( "setlog" );
        REQUIRE( sets.size() == 6 );
        for( size_t i = 0; i < sets.size(); ++i )
        {
            REQUIRE( sets[i] == "-s " + wheel + ".filterName." + ( i % 2 == 0 ? "ND3_0" : "ND1_0" ) + "=On" );
        }
        REQUIRE( indi.get( wheel + ".filterName.ND1_0" ) == "On" );
    }

    SECTION( "a run ending in the unsaturated segment puts the wheel back" )
    {
        std::vector<std::string> a = { "--psf.wheel=" + wheel, "--psf.filter=ND3_0", "--psf.unsatLast=5" };
        result                   r = runApp( "allunsat", a );
        REQUIRE( r.rv == 0 );
        REQUIRE( r.nDone == 2 );
        mx::improc::eigenCube<float> ref;
        mx::fits::fitsHeader<>       head;
        REQUIRE( ff.read( ref, head, ( dir / ( "allunsat_" + cfg.sciName + "_ref.fits" ) ).string() ) ==
                 mx::error_t::noerror );
        REQUIRE( head["NREFUNS"].value<int>() == head["NREF"].value<int>() );
        for( int p = 0; p < ref.planes(); ++p )
        {
            REQUIRE( ref.image( p ).maxCoeff() < 37000 );
        }
        REQUIRE( indi.lines( "setlog" ).size() == 2 );
        REQUIRE( indi.get( wheel + ".filterName.ND1_0" ) == "On" );
    }

    SECTION( "an unsaturated segment needs the wheel away from the PSF filter" )
    {
        indi.set( wheel + ".filterName.ND1_0", "Off" );
        indi.set( wheel + ".filterName.ND3_0", "On" );
        result r = runApp( "splitat", { "--psf.wheel=" + wheel, "--psf.filter=ND3_0", "--psf.unsatLast=0" } );
        REQUIRE( r.rv == -1 );
        REQUIRE( r.nDone == 0 );
        REQUIRE( indi.lines( "setlog" ).empty() );
    }

    SECTION( "calibration speckles need a PSF block" )
    {
        result r = runApp( "nopsf", { "--psf.speckAmp=0.5" } );
        REQUIRE( r.rv == -1 );
        REQUIRE( r.nDone == 0 );
    }

    SECTION( "refuses to saturate with EM gain above 1" )
    {
        indi.set( dev + ".emgain.current", "10" );
        result r = runApp( "emgain", psfArgs );
        REQUIRE( r.rv == -1 );
        REQUIRE( r.nDone == 0 );
        REQUIRE( indi.lines( "setlog" ).empty() );
    }

    SECTION( "a PSF needs indi.enable" )
    {
        std::vector<std::string> a = psfArgs;
        a.push_back( "--indi.enable=false" );
        result r = runApp( "noindi", a );
        REQUIRE( r.rv == -1 );
        REQUIRE( indi.lines( "getlog" ).empty() );
    }

    sim.stop();
    std::filesystem::remove_all( dir );
}
