ensure_path( 'TEXINPUTS', './tex//' );
ensure_path( 'BIBINPUTS', './tex//' );

# Path to your Python virtual environment
my $env_path = 'env';
$ENV{'PATH'} = "$env_path/bin:" . $ENV{'PATH'};
$ENV{'VIRTUAL_ENV'} = $env_path; # Use python environment

$ENV{'JULIA_PROJECT'} = '.'; # Set JULIA_PROJECT environment variable

$pdf_mode = 4;
$out_dir = 'output';
$aux_dir = 'output';

$MSWin_back_slash = 0;

$clean_ext .= " pythontex-files-%R/* pythontex-files-%R";
push @generated_exts, 'pytxcode';

$pythontex = 'pythontex --interpreter python:/usr/bin/python3 %O %S';
$extra_rule_spec{'pythontex'}  = [ 'internal', '', 'mypythontex', "%Y%R.pytxcode",  "%Ypythontex-files-%R/%R.pytxmcr",    "%R", 1 ];

sub mypythontex {
   my $result_dir = $aux_dir1."pythontex-files-$$Pbase";
   my $ret = Run_subst( $pythontex, 2 );
   rdb_add_generated( glob "$result_dir/*" );
   my $fh = new FileHandle $$Pdest, "r";
   if ($fh) {
      while (<$fh>) {
         if ( /^%PythonTeX dependency:\s+'([^']+)';/ ) {
	     print "Found pythontex dependency '$1'\n";
             rdb_ensure_file( $rule, $aux_dir1.$1 );
	 }
      }
      undef $fh;
   }
   else {
       warn "mypythontex: I could not read '$$Pdest'\n",
            "  to check dependencies\n";
   }
   return $ret;
}

sub pull_julia_code {
   system("julia --project=. --color=yes jl/pull_julia_code.jl");
}

pull_julia_code();

$lualatex = "lualatex %O -shell-escape %S";

@default_files = ('main');