# ANNOTATED SPECFILE FOR hellocopr EXAMPLE PROGRAM

#-- OWN DEFINITIONS/CONDITIONALS ----------------------------------------------#
# Not necessary, but doing these definitions at the top of
# the specfile helps readability

# You can use the %%global macro to define your own variables
# can be paired with %%if structures to set up switches
%global giturl https://github.com/DaveDuck321/GrapheneLang

#-- PREAMBLE ------------------------------------------------------------------#
# This will be the name of the package created. Needs to match
# the name of the specfile
Name: glang
# Version as given by upstream
Version: 0.0.0
# Iteration of the the package of this particular version
# Increase when you change the package without changing the
# version. Always append %%{?dist}, this takes care of adding
# the e.g. .f32 to the package
Release: 0%{?dist}
# Multiple licenses can be combined with logical operators, e.g.
# GPLv3 and MIT and LGPL
# If different parts of the code use different licenses, you should
# add a comment here to clarify which license covers what
License: AGPL-3.0-only
# A short description of less than 80 characters
Summary: A Graphene front-end for LLVM
# Upstream URL
Url: %{giturl}/
# URL where to obtain the sources
Source0: https://github.com/DaveDuck321/GrapheneLang/archive/refs/heads/main.zip
# Source0: %{giturl}/archive/refs/tags/%{version}.tar.gz
# You can add multiple source files by adding this more than once,
# appending a new number, i.e. SourceN. These values are available
# later in the %%{SOURCE0}, %%{SOURCE1}, ... macros.
# Source1: myconfig.conf

# Add patches you want to apply to the sources as PatchN:, these
# will be automatically applied by RPM later
# Patch0: mypatch.patch

# Which arch the package is to be built for. Mainly useful to mark
# arch-less packages as such
# BuildArch: noarch

# List of packages required for building this package
BuildRequires: python3-devel
BuildRequires: python3-setuptools

# List of packages required by the package at runtime
# Requires: ...
# Here, the RPM python macros automatically take care of the dependencies,
# so none are listed

# Full description of the package
# Can be multiline, anything until the next section (%%prep) becomes part of
# the description text. Wrap at 80 characters.
%description
Hellocopr is a very simple demonstration program that does nothing but display
some text on the command line. It is used as an example for automatic RPM
packaging using tito and Fedora's Copr user repository.

#-- PREP, BUILD & INSTALL -----------------------------------------------------#
# The %%prep stage is used to set up the build directories, unpack & copy the
# sources, apply patches etc.. Basically anything that needs to be done before
# running ./configure in the usual ./configure, make, make install workflow
%prep
# often, the %%autosetup macro is all that is needed here. This will unpack &
# copy the sources to the correct directories and apply patches. If your source
# tarball does not extract to a directory of the same name, you can specify
# the directory using the -n <dir> switch. You can also pass the -p option of
# the patch utility
%autosetup -n GrapheneLang-main

# The %%build stage is used to build the software. Most common build commands
# have macros that take care of setting the appropriate environment, directories,
# flags, etc., so for './configure', you'd use %%{configure}, for 'make' %%{make_build},
# for 'python setup.py build' %%{py3_build} etc.
# This stage contains everything that needs to be done in the source directory before
# installing the software on a target system
%build
# %%py3_build

# the %%install stage is used to install the software. This
# uses the actual installation paths using %%{buildroot} as the root, i.e.
# %%{buildroot}/usr/share becomes /usr/share when the package is installed on
# a real system.
# There are RPM macros for most standard paths (e.g. %%{_sysconfdir} for /etc,
# %%{_bindir} for /usr/bin and so on), try to use those instead of hardcoding
# the paths. This avoids errors and makes it easier to adapt to filesystem
# changes
%install
%py3_install

#-- FILES ---------------------------------------------------------------------#
# The files section list every file contained in the package, pretty much
# the list of files created in the %%install section. There are a number of
# special flags, like %%doc, %%license or %%dir that tell RPM what kind of
# file it is dealing with.
%files
# %%doc README.md
%license LICENSE
%{_bindir}/hellocopr
%{python3_sitelib}/%{name}-*.egg-info/
%{python3_sitelib}/%{name}/

#-- CHANGELOG -----------------------------------------------------------------#
# The changelog is the last section of the specfile. Everything after this is
# treated as part of the changelog.
# Entries should follow the format given below and be separated by one empty line.
# A * marks the beginning of a new entry and is followed by date, author and package
# version. Lines beginning with - after that list the changes contained in the
# package.
%changelog
* Fri Jul 24 2020 Christopher Engelhard <ce@lcts.de> 1.0.2-1
- let tito manage the version string ( ce@lcts.de)

* Fri Jul 24 2020 Christopher Engelhard <ce@lcts.de> 1.0.1-1
- single-source program version (ce@lcts.de)

* Fri Jul 24 2020 Christopher Engelhard <ce@lcts.de> 1.0.0-1
- new package built with tito
